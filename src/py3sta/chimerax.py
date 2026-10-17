"""Send commands to a running ChimeraX through its REST remote control.

In ChimeraX: ``remotecontrol rest start port 60000`` (add ``json true`` for structured replies; both work).
ChimeraX must run where the files are (or see the same paths); for ChimeraX on a laptop use a reverse tunnel
(``ssh -R 60000:localhost:60000 workstation``) and a shared filesystem.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_PORT = 60000
# ChimeraX 1.10 'json true' replies to 'open' with this error after opening the map (the returned Volume cannot be
# encoded); the command itself succeeded
SERIALIZE_BUG = "object has no attribute 'string'"
START_HINT = "in ChimeraX run: remotecontrol rest start port {port}"


class ChimeraXError(RuntimeError):
    pass


def run(command: str, port: int = DEFAULT_PORT, host: str = "127.0.0.1", timeout: float = 30.0) -> str:
    """Run one ChimeraX command; returns its log text. Raises ChimeraXError if ChimeraX is not reachable or
    reports an error."""
    url = f"http://{host}:{port}/run?" + urllib.parse.urlencode({"command": command})
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read().decode(errors="replace")
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        raise ChimeraXError(f"ChimeraX not reachable on {host}:{port} ({e}); " + START_HINT.format(port=port))
    try:
        reply = json.loads(body)                       # remotecontrol rest ... json true
    except ValueError:
        return body
    logs = reply.get("log messages") or {}
    text = "\n".join(m.rstrip() for level in logs.values() for m in level) if isinstance(logs, dict) else str(logs)
    err = reply.get("error")
    if err:
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        if SERIALIZE_BUG in msg and text:              # ChimeraX ran the command but cannot encode its result
            return text
        raise ChimeraXError(msg)
    return text


def open_command(path, name: str | None = None) -> str:
    """ChimeraX command that opens an MRC map or stack (``.mrcs`` as volume series, ``format mrc``)."""
    p = Path(path).resolve()
    cmd = f'open "{p}" format mrc'
    if name:
        cmd += f' name "{name}"'
    return cmd


def stack_count(path) -> tuple[int, int]:
    """(volumes, box) of an MRC file: a stack of k cubic maps is stored as (k*box, box, box)."""
    from .io.mrc import read_header

    z, y, x = read_header(path)["stored_shape"]
    return (z // y, y) if y == x and z % y == 0 and z > y else (1, y)


def open_maps(paths, port: int = DEFAULT_PORT, host: str = "127.0.0.1", gap: float = 1.25) -> str:
    """Open map file(s) in ChimeraX; returns ChimeraX's log. A stack (class averages, eigenvolumes) becomes one
    model per volume, side by side along x (``gap`` box widths apart), named ``file [i]``."""
    import re

    out = []
    for p in paths:
        k, box = stack_count(p)
        log = run(open_command(p, Path(p).name), port, host)
        out.append(log)
        if k == 1:
            continue
        m = re.search(r" as #([\d.]+)", log)
        if not m:
            raise ChimeraXError(f"could not find the model number of {Path(p).name} in: {log}")
        src = m.group(1)
        for i in range(k):
            log = run(f"volume copy #{src} subregion 0,0,{i * box},{box - 1},{box - 1},{(i + 1) * box - 1}", port, host)
            new = re.search(r" as #([\d.]+)", log).group(1)
            run(f"volume #{new} originIndex {box / 2 - i * gap * box:g},{box / 2:g},{box / 2:g}", port, host)
            run(f'rename #{new} "{Path(p).name} [{i + 1}]"', port, host)
        run(f"close #{src}", port, host)
        out.append(f"{Path(p).name}: {k} volumes as separate models, side by side along x")
    return "\n".join(out)
