"""Step manifests, change tracking and step states (PLAN.md §1 "Staleness and change tracking").

Each step folder holds ``manifest.json``: parameters, inputs (STAR fingerprints, external file
size/mtime, upstream run ids), outputs, run id, versions, device, timing and status.
A step is *stale* when its parameters, inputs or upstream runs differ from what it recorded.
STAR files may be edited by hand: an edited output is reported (``edited``), protected from being
overwritten without ``force`` (a backup is kept), and makes the steps that read it stale.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .. import __version__

MANIFEST = "manifest.json"


class StepError(RuntimeError):
    """A step cannot run (stale upstream, edited outputs, ...) or failed."""


# ------------------------------------------------------------ fingerprints
def sha1(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def file_stat(path) -> dict:
    st = os.stat(path)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def fingerprint(path) -> dict:
    return {**file_stat(path), "sha1": sha1(path)}


def same_content(fp: dict, path) -> bool:
    """True if ``path`` still has the content of fingerprint ``fp``. The hash is only computed when
    size or mtime differ, so ``touch`` or a copy is not a change but an edit is."""
    try:
        st = file_stat(path)
    except FileNotFoundError:
        return False
    if st["size"] != fp["size"]:
        return False
    return st["mtime_ns"] == fp["mtime_ns"] or sha1(path) == fp["sha1"]


def same_stat(rec: dict, path) -> bool:
    try:
        st = file_stat(path)
    except FileNotFoundError:
        return False
    return st["size"] == rec["size"] and st["mtime_ns"] == rec["mtime_ns"]


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode:
            return None
        dirty = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def read_manifest(folder) -> dict | None:
    p = Path(folder) / MANIFEST
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def write_manifest(folder, m: dict) -> None:
    p = Path(folder) / MANIFEST
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=1, default=str))
    os.replace(tmp, p)


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------ step context
class StepContext:
    """Handed to a step's run function; records what the step reads and writes."""

    def __init__(self, root: Path, folder: Path, manifest: dict, log=None):
        self.root, self.folder, self.manifest = root, folder, manifest
        self.log = log or (lambda msg: None)

    def _rel(self, path) -> str:
        p = Path(path).absolute()
        try:
            return str(p.relative_to(self.root.absolute()))
        except ValueError:
            return str(p)

    def read_star(self, path) -> Path:
        """Record a STAR input (fingerprint, so hand edits are detected)."""
        self.manifest["inputs"]["star"][self._rel(path)] = fingerprint(path)
        return Path(path)

    def use_files(self, paths, kind: str) -> None:
        """Record external inputs (tomograms, XMLs, MRCs) by size + mtime."""
        for p in paths:
            self.manifest["inputs"]["files"][os.path.abspath(p)] = {**file_stat(p), "kind": kind}

    def use_step(self, name: str, run_id: str) -> None:
        self.manifest["inputs"]["steps"][name] = run_id

    def output(self, name: str) -> Path:
        """Path of an output file in the step folder (removed when the step is re-run)."""
        self.manifest["outputs"]["files"].append(name)
        return self.folder / name

    def output_star(self, name: str) -> Path:
        """Path of a STAR output; fingerprinted when the step finishes (hand edits are tracked)."""
        self.manifest["outputs"]["star"][name] = None
        return self.folder / name

    @property
    def info(self) -> dict:
        return self.manifest["info"]

    def warn(self, msg: str) -> None:
        self.manifest["warnings"].append(msg)
        self.log(f"warning: {msg}")


LOG_FILE = "log.txt"


def _tee(path: Path, log):
    """A log function that also appends time-stamped lines to ``path`` (flushed: readable while running)."""
    fh = open(path, "w", buffering=1)

    def write(msg: str) -> None:
        fh.write(f"{dt.datetime.now():%H:%M:%S} {msg}\n")
        if log is not None:
            log(msg)
    return write, fh


def run_step(root, folder, name: str, params: dict, fn, force: bool = False, device: str | None = None,
             log=None, extra: dict | None = None) -> dict:
    """Run ``fn(ctx)`` as step ``name`` in ``folder`` with change tracking; returns the manifest.
    The step's log lines are also written to ``folder/log.txt``.

    Refuses if a previous run is still running, or if one of its STAR outputs was edited (``force``:
    the edited file is kept as ``<name>.edited-<date>.star``). Old outputs are removed first.
    Fails if an external input changed during the run.
    """
    root, folder = Path(root), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    old = read_manifest(folder)
    if old:
        if old["status"] == "running" and _alive(old):
            raise StepError(f"{folder.name} is already running (pid {old['pid']} on {old['host']})")
        edited = [n for n, fp in old["outputs"]["star"].items() if fp and not same_content(fp, folder / n)]
        if edited and not force:
            raise StepError(f"{folder.name}: {', '.join(edited)} edited by hand; re-running would overwrite it "
                            "(use --force; the edited version is kept as a backup)")
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        for n in edited:
            p = folder / n
            if p.exists():
                shutil.copyfile(p, p.with_name(f"{p.stem}.edited-{stamp}{p.suffix}"))
        for n in list(old["outputs"]["files"]) + list(old["outputs"]["star"]):
            (folder / n).unlink(missing_ok=True)
    m = {"step": name, "run_id": uuid.uuid4().hex[:12], "status": "running", "started": _now(),
         "finished": None, "seconds": None, "host": socket.gethostname(), "pid": os.getpid(),
         "py3sta": __version__, "git": git_commit(), "device": device, "params": params,
         "inputs": {"star": {}, "files": {}, "steps": {}}, "outputs": {"star": {}, "files": []},
         "info": {}, "warnings": [], "error": None, **(extra or {})}
    write_manifest(folder, m)
    log, log_fh = _tee(folder / LOG_FILE, log)
    ctx = StepContext(root, folder, m, log)
    t0 = time.time()
    try:
        fn(ctx)
        changed = [p for p, rec in m["inputs"]["files"].items() if not same_stat(rec, p)]
        if changed:
            raise StepError(f"input changed during the run: {', '.join(changed[:3])}")
        for n in m["outputs"]["star"]:
            m["outputs"]["star"][n] = fingerprint(folder / n)
        m["status"] = "done"
    except BaseException as e:
        m["status"] = "failed"
        m["error"] = f"{type(e).__name__}: {e}"
        m["traceback"] = traceback.format_exc(limit=8)
        log_fh.write(f"{m['error']}\n{m['traceback']}")
        raise
    finally:
        m["finished"] = _now()
        m["seconds"] = round(time.time() - t0, 2)
        write_manifest(folder, m)
        log_fh.close()
    return m


def _alive(m: dict) -> bool:
    if m.get("host") != socket.gethostname():
        return True                                  # cannot check another host: assume running
    try:
        os.kill(int(m["pid"]), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --------------------------------------------------------------- states
@dataclass
class StepState:
    state: str                                     # missing | running | interrupted | failed | stale | done
    reasons: list = field(default_factory=list)    # why stale / failed
    edited: list = field(default_factory=list)     # STAR outputs edited by hand
    manifest: dict | None = None

    @property
    def ok(self) -> bool:
        return self.state == "done"

    def describe(self) -> str:
        s = self.state + (f" (edited: {', '.join(self.edited)})" if self.edited else "")
        return s + (": " + "; ".join(self.reasons) if self.reasons else "")


def diff_params(old, new, prefix: str = "") -> list[str]:
    """Human-readable differences between two parameter dicts: 'ctf.mode none→flip'."""
    out = []
    for k in sorted(set(old) | set(new), key=str):
        a, b = old.get(k, "<unset>"), new.get(k, "<unset>")
        if isinstance(a, dict) and isinstance(b, dict):
            out += diff_params(a, b, f"{prefix}{k}.")
        elif a != b:
            out.append(f"{prefix}{k} {a}→{b}")
    return out


def check_state(root, folder, params: dict | None, upstream_state=None) -> StepState:
    """State of the step in ``folder``. ``params``: current parameters (None = do not compare).
    ``upstream_state(name) -> StepState`` resolves upstream steps recorded in the manifest."""
    root, folder = Path(root), Path(folder)
    m = read_manifest(folder)
    if m is None:
        return StepState("missing")
    if m["status"] == "running":
        return StepState("running" if _alive(m) else "interrupted", manifest=m)
    if m["status"] == "failed":
        return StepState("failed", [m.get("error") or "unknown error"], manifest=m)
    reasons = []
    if params is not None:
        reasons += diff_params(json.loads(json.dumps(m["params"], default=str)),
                               json.loads(json.dumps(params, default=str)))
    for rel, fp in m["inputs"]["star"].items():
        p = Path(rel) if os.path.isabs(rel) else root / rel
        if not same_content(fp, p):
            reasons.append(f"{rel} {'missing' if not p.exists() else 'changed'}")
    changed = {}
    for p, rec in m["inputs"]["files"].items():
        if not same_stat(rec, p):
            changed.setdefault(rec.get("kind", "file"), []).append(Path(p).name)
    for kind, names in changed.items():
        reasons.append(f"{len(names)} {kind}(s) changed: {', '.join(names[:3])}" + (" ..." if len(names) > 3 else ""))
    if upstream_state is not None:
        for up, rid in m["inputs"]["steps"].items():
            us = upstream_state(up)
            if us.manifest is None or us.manifest.get("run_id") != rid:
                reasons.append(f"upstream {up} re-run" if us.manifest else f"upstream {up} {us.state}")
            elif not us.ok:
                reasons.append(f"upstream {up} {us.state}")
    edited = [n for n, fp in m["outputs"]["star"].items() if fp and not same_content(fp, folder / n)]
    return StepState("stale" if reasons else "done", reasons, edited, m)
