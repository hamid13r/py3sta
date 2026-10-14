"""Minimal MRC2014 reader/writer (numpy only). Arrays are indexed [z, y, x]."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_MODES = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16, 12: np.float16}
_MODE_OF = {np.dtype(v): k for k, v in _MODES.items()}


def read_header(path) -> dict:
    with open(path, "rb") as f:
        h = f.read(1024)
    i = np.frombuffer(h, "<i4")
    fl = np.frombuffer(h, "<f4")
    nx, ny, nz, mode = (int(v) for v in i[:4])
    mx = int(i[7]) or nx
    if tuple(i[16:19]) not in ((1, 2, 3), (0, 0, 0)):
        raise NotImplementedError(f"{path}: non-standard axis order MAPC/R/S={tuple(i[16:19])}")
    if mode not in _MODES:
        raise NotImplementedError(f"{path}: MRC mode {mode} not supported")
    return {"shape": (nz, ny, nx), "mode": mode, "dtype": _MODES[mode],
            "apix": float(fl[10]) / mx if fl[10] else 1.0,
            "offset": 1024 + int(i[23]), "origin": tuple(float(v) for v in fl[49:52])}


def read_mrc(path, mmap: bool = True):
    """Return (array[z,y,x], apix). With mmap=True the file is not loaded into RAM."""
    h = read_header(path)
    if mmap:
        a = np.memmap(path, dtype=h["dtype"], mode="r", offset=h["offset"], shape=h["shape"])
    else:
        a = np.fromfile(path, dtype=h["dtype"], offset=h["offset"],
                        count=int(np.prod(h["shape"]))).reshape(h["shape"])
    return a, h["apix"]


def write_mrc(path, data, apix: float = 1.0, origin=(0.0, 0.0, 0.0)) -> None:
    data = np.ascontiguousarray(data)
    if data.dtype not in _MODE_OF:
        data = data.astype(np.float32)
    nz, ny, nx = data.shape
    hi = np.zeros(256, "<i4")
    hf = hi.view("<f4")
    hi[:4] = (nx, ny, nz, _MODE_OF[data.dtype])
    hi[7:10] = (nx, ny, nz)
    hf[10:13] = (nx * apix, ny * apix, nz * apix)
    hf[13:16] = 90.0
    hi[16:19] = (1, 2, 3)
    d = data.astype(np.float64)
    hf[19:22] = (d.min(), d.max(), d.mean())
    hf[49:52] = origin
    hb = bytearray(hi.tobytes())
    hb[208:212] = b"MAP "
    hb[212:216] = bytes([0x44, 0x44, 0, 0])  # little-endian machine stamp
    hi2 = np.frombuffer(bytes(hb), "<i4").copy()
    hi2[27] = 20140  # nversion
    hf2 = hi2.view("<f4")
    hf2[54] = d.std()  # rms
    with open(path, "wb") as f:
        f.write(hi2.tobytes())
        f.write(data.tobytes())
