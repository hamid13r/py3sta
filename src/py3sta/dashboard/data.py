"""What the dashboard shows, read from the project folder through the pipeline API (no Dash here).

The project folder is the single source of truth: every function reads the current files, so CLI runs show up
on the next refresh. Functions return plain Python / numpy values; the pages only render them.
"""
from __future__ import annotations

import base64
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..pipeline.manifest import LOG_FILE, read_manifest
from ..pipeline.runner import project_status
from ..pipeline.steps import STEPS, get_step

AVERAGE = ("average", "average.mrc")                     # (step, file) of the cycle average


@dataclass
class StepCell:
    name: str
    folder: str
    state: str                                           # missing | running | interrupted | failed | stale | done
    reasons: list = field(default_factory=list)
    edited: list = field(default_factory=list)
    seconds: float | None = None
    finished: str | None = None
    device: str | None = None
    n_warnings: int = 0
    error: str | None = None


@dataclass
class CycleRow:
    n: int
    n_particles: int | None
    n_excluded: int | None
    ctf_mode: str | None
    steps: list                                          # [StepCell] in pipeline order
    fsc143: float | None = None                          # first crossing (Å), 01_average
    fsc143_last: float | None = None
    n_used: int | None = None
    error: str | None = None


@dataclass
class Overview:
    root: str
    name: str
    cycles: list                                         # [CycleRow]
    stack: str                                           # one-line stack summary


def overview(project) -> Overview:
    """One row per cycle: particle counts, the state of each implemented step, the average's resolution."""
    st = project_status(project)
    rows = []
    for cs in st.cycles:
        cells = []
        for s, state in cs.steps:
            if state.state == "not implemented":
                continue
            m = state.manifest or {}
            cells.append(StepCell(s.name, s.folder, state.state, list(state.reasons), list(state.edited),
                                  m.get("seconds"), m.get("finished"), m.get("device"), len(m.get("warnings") or []),
                                  m.get("error")))
        row = CycleRow(cs.n, cs.n_particles, cs.n_excluded, cs.ctf_mode, cells, error=cs.error)
        avg = read_manifest(project.cycle_dir(cs.n) / get_step("average").folder)
        if avg and avg.get("status") == "done":
            info = avg.get("info", {})
            row.fsc143, row.fsc143_last, row.n_used = info.get("fsc143_first"), info.get("fsc143_last"), info.get("n_used")
        rows.append(row)
    if st.stack_rows:
        stack = f"stack: {st.stack_rows} particles, {st.stack_bytes / 1e9:.1f} GB"
        if st.drift_warning:
            stack += f"; {st.drift_warning} particle(s) beyond half the drift allowance"
    else:
        stack = "stack: not built yet"
    return Overview(str(project.root), project.root.name, rows, stack)


def resolution_history(ov: Overview) -> list[tuple[int, float, float | None, int | None]]:
    """[(cycle, FSC 0.143 first crossing, last crossing, particles used)] for cycles with a finished average."""
    return [(r.n, r.fsc143, r.fsc143_last, r.n_used) for r in ov.cycles if r.fsc143 is not None]


def step_folder(project, n: int, step: str) -> Path:
    return project.cycle_dir(n) / get_step(step).folder


def step_log(project, n: int, step: str) -> dict:
    """{'log': text of log.txt ('' if none), 'warnings': [...], 'error': str|None, 'traceback': str|None,
    'status': ..., 'manifest': bool}."""
    folder = step_folder(project, n, step)
    m = read_manifest(folder) or {}
    p = folder / LOG_FILE
    return {"log": p.read_text(errors="replace") if p.is_file() else "", "warnings": list(m.get("warnings") or []),
            "error": m.get("error"), "traceback": m.get("traceback"), "status": m.get("status"),
            "manifest": bool(m)}


def fsc_curve(project, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(resolution Å, FSC) of the cycle's half-map FSC (01_average/fsc.txt), or None."""
    p = step_folder(project, n, "average") / "fsc.txt"
    if not p.is_file():
        return None
    a = np.loadtxt(p, comments="#", ndmin=2)
    return a[:, 0], a[:, 1]


def cycle_average(project, n: int) -> Path | None:
    p = step_folder(project, n, AVERAGE[0]) / AVERAGE[1]
    return p if p.is_file() else None


# ------------------------------------------------------------------ images
def view(vol: np.ndarray, axis: int = 0, index: int | None = None, project: bool = False) -> np.ndarray:
    """2D view of a (z, y, x) volume: the projection along ``axis`` or the slice ``index`` (default: centre).
    axis 0 = XY (looking down z), 1 = XZ, 2 = YZ."""
    vol = np.asarray(vol, np.float32)
    if project:
        return vol.sum(axis)
    i = vol.shape[axis] // 2 if index is None else int(np.clip(index, 0, vol.shape[axis] - 1))
    return np.take(vol, i, axis=axis)


def to_uint8(img: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """Percentile contrast stretch to 0..255."""
    img = np.asarray(img, np.float64)
    a, b = np.percentile(img, [lo, hi])
    if b <= a:
        return np.zeros(img.shape, np.uint8)
    return (np.clip((img - a) / (b - a), 0, 1) * 255).round().astype(np.uint8)


def png_bytes(gray: np.ndarray) -> bytes:
    """Grayscale uint8 image (row 0 at the top) -> PNG (no imaging library needed)."""
    gray = np.ascontiguousarray(gray, np.uint8)
    h, w = gray.shape
    raw = b"".join(b"\x00" + gray[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def png_data_uri(gray: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(gray)).decode()


def thumbnail(path, project_views: bool = True) -> str:
    """Data URI of the three orthogonal projections (or central slices) of a map side by side; y up."""
    from ..io.mrc import read_mrc

    vol = np.asarray(read_mrc(path, mmap=False)[0], np.float32)
    if vol.ndim == 4:
        vol = vol[0]
    views = [to_uint8(view(vol, a, project=project_views))[::-1] for a in range(3)]
    gap = np.full((views[0].shape[0], 2), 255, np.uint8)
    h = max(v.shape[0] for v in views)
    views = [np.pad(v, ((0, h - v.shape[0]), (0, 0)), constant_values=255) for v in views]
    return png_data_uri(np.hstack([views[0], gap, views[1], gap, views[2]]))


STEP_NAMES = [s.name for s in STEPS]


# ------------------------------------------------------------------ maps
MAP_SUFFIXES = (".mrc", ".mrcs", ".map")


def cycle_maps(project, n: int) -> list[str]:
    """Maps of a cycle (paths relative to the cycle folder), in step order."""
    d = project.cycle_dir(n)
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.suffix in MAP_SUFFIXES and p.is_file())


_VOL_CACHE: dict = {}


def load_volumes(path, box: int | None = None) -> np.ndarray:
    """(k, z, y, x) float32: one map, or a stack of k cubic maps stored as (k*box, box, box). Cached by mtime."""
    from ..io.mrc import read_mrc

    path = Path(path)
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    if key not in _VOL_CACHE:
        a = np.asarray(read_mrc(path, mmap=False)[0], np.float32)
        if a.ndim == 4:
            pass
        elif a.ndim == 3 and a.shape[1] == a.shape[2] and a.shape[0] % a.shape[1] == 0 and a.shape[0] > a.shape[1]:
            a = a.reshape(-1, a.shape[1], a.shape[1], a.shape[1])
        elif a.ndim == 3:
            a = a[None]
        else:
            raise ValueError(f"{path.name}: not a 3D map (shape {a.shape})")
        if len(_VOL_CACHE) > 32:
            _VOL_CACHE.clear()
        _VOL_CACHE[key] = a
    return _VOL_CACHE[key]


def filtered(vol: np.ndarray, apix: float, lowpass: float | None = None) -> np.ndarray:
    """Low-passed copy (Å) of a cubic map; unchanged if lowpass is None or beyond Nyquist."""
    from ..reference import bandpass

    if not lowpass or lowpass <= 2 * apix or len(set(vol.shape)) != 1:
        return vol
    return np.asarray(bandpass(vol, apix, lowpass=lowpass), np.float32)


def ortho_views(vol: np.ndarray, indices=(None, None, None), project_views: bool = False) -> list[np.ndarray]:
    """[XY, XZ, YZ] views: slices at ``indices`` (z, y, x; None = centre) or projections."""
    return [view(vol, a, indices[a], project_views) for a in range(3)]


def cycle_pngs(project, n: int) -> list[str]:
    """Diagnostic PNGs of a cycle (relative to the cycle folder), in step order."""
    d = project.cycle_dir(n)
    return sorted(str(p.relative_to(d)) for p in d.rglob("*.png") if p.is_file())


def all_fsc(project) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {n: c for n in project.cycles if (c := fsc_curve(project, n)) is not None}


def star_table(path, block: str | None = None):
    """A STAR block as a DataFrame (first block if ``block`` is None), or None if the file is missing."""
    from ..io.star import read_star

    path = Path(path)
    if not path.is_file():
        return None
    blocks = read_star(path)
    return blocks[block] if block else next(iter(blocks.values()))


def scree(project, n: int):
    return star_table(step_folder(project, n, "msa") / "scree.star")


def classify_compare(project, n: int):
    return star_table(step_folder(project, n, "classify") / "compare.star")


def project_file(project, rel: str) -> Path:
    """Path of a file inside the project folder; refuses paths that leave it."""
    root = project.root.resolve()
    p = (root / rel).resolve()
    if root not in p.parents:
        raise ValueError(f"{rel}: outside the project")
    return p
