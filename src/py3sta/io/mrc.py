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


def header_bytes(shape, dtype, apix: float = 1.0, origin=(0.0, 0.0, 0.0), stats=(0.0, 0.0, 0.0, 0.0)) -> bytes:
    """1024-byte MRC2014 header for an array of ``shape`` [z, y, x]; stats = (min, max, mean, rms)."""
    nz, ny, nx = shape
    hi = np.zeros(256, "<i4")
    hf = hi.view("<f4")
    hi[:4] = (nx, ny, nz, _MODE_OF[np.dtype(dtype)])
    hi[7:10] = (nx, ny, nz)
    hf[10:13] = (nx * apix, ny * apix, nz * apix)
    hf[13:16] = 90.0
    hi[16:19] = (1, 2, 3)
    hf[19:22] = stats[:3]
    hf[49:52] = origin
    hi[27] = 20140  # nversion
    hf[54] = stats[3]
    hb = bytearray(hi.tobytes())
    hb[208:212] = b"MAP "
    hb[212:216] = bytes([0x44, 0x44, 0, 0])  # little-endian machine stamp
    return bytes(hb)


def write_mrc(path, data, apix: float = 1.0, origin=(0.0, 0.0, 0.0)) -> None:
    data = np.ascontiguousarray(data)
    if data.dtype not in _MODE_OF:
        data = data.astype(np.float32)
    d = data.astype(np.float64)
    with open(path, "wb") as f:
        f.write(header_bytes(data.shape, data.dtype, apix, origin, (d.min(), d.max(), d.mean(), d.std())))
        f.write(data.tobytes())


def open_volume_stack(path, n: int, size: int, dtype=np.float16, apix: float = 1.0, mode: str = "r+"):
    """Stack of n cubes (size^3) stored as one MRC of shape (n*size, size, size); returns memmap (n, size, size, size).

    mode "w+" creates (or truncates) the file; "r+" / "r" open an existing one. ``resize_volume_stack`` grows it.
    """
    dtype = np.dtype(dtype)
    if mode == "w+":
        with open(path, "wb") as f:
            f.write(header_bytes((n * size, size, size), dtype, apix))
            f.truncate(1024 + n * size ** 3 * dtype.itemsize)
    h = read_header(path)
    nz, ny, nx = h["shape"]
    if ny != size or nx != size or nz % size:
        raise ValueError(f"{path}: shape {h['shape']} is not a stack of {size}^3 cubes")
    return np.memmap(path, dtype=h["dtype"], mode="r" if mode == "r" else "r+", offset=h["offset"],
                     shape=(nz // size, size, size, size))


def resize_volume_stack(path, n: int, size: int) -> None:
    """Grow (or shrink) a volume stack to n cubes in place; new cubes are zero."""
    h = read_header(path)
    dtype = np.dtype(h["dtype"])
    with open(path, "r+b") as f:
        f.write(header_bytes((n * size, size, size), dtype, h["apix"]))
        f.truncate(1024 + n * size ** 3 * dtype.itemsize)
