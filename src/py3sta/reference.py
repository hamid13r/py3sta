"""References: masks, band-pass filters and reference preparation (PLAN.md §2 step 2).

All physical sizes in Å; volumes are [z, y, x] with the centre at voxel box//2 (as average.box_grid).
Soft edges are raised cosines centred on the nominal size: a sphere of radius R with soft edge s is 1
inside R - s/2, 0 outside R + s/2 and 0.5 at R.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .average import box_grid
from .io.mrc import read_header, read_mrc

EDGE_SHELLS = 3          # width of the band-pass raised-cosine edges, in Fourier shells
MASK_SHAPES = ("sphere", "ellipsoid", "box", "file")


def _falloff(d, width):
    """1 for d <= -width/2, 0 for d >= width/2, raised cosine in between (d = distance outside the edge)."""
    if width <= 0:
        return (d <= 0).astype(np.float32)
    t = np.clip(d / width + 0.5, 0, 1)
    return (0.5 + 0.5 * np.cos(np.pi * t)).astype(np.float32)


def make_mask(box: int, apix: float, shape: str = "sphere", radius=(150.0,), soft: float = 20.0,
              center=(0.0, 0.0, 0.0), file=None) -> np.ndarray:
    """Soft mask in [0, 1]. radius: 1 value (sphere) or 3 (x, y, z: ellipsoid radii / box half-sizes), Å.
    center: offset of the mask centre from the box centre (x, y, z), Å. file: MRC with the same box and
    voxel size (values clipped to [0, 1])."""
    if shape == "file":
        if not file:
            raise ValueError("mask shape 'file' needs a file")
        vol, a = read_mrc(file, mmap=False)
        if vol.shape != (box,) * 3:
            raise ValueError(f"mask {file}: shape {vol.shape} differs from the box {box}")
        if abs(a - apix) > 1e-3 * apix:
            raise ValueError(f"mask {file}: voxel size {a:g} Å differs from {apix:g} Å")
        return np.clip(vol, 0, 1).astype(np.float32)
    if shape not in MASK_SHAPES:
        raise ValueError(f"unknown mask shape {shape!r}")
    radius = np.atleast_1d(np.asarray(radius, float))
    x = box_grid(box) * apix - np.asarray(center, np.float32)        # Å from the mask centre, xyz
    if shape == "sphere":
        return _falloff(np.sqrt((x ** 2).sum(-1)) - radius[0], soft)
    if len(radius) != 3:
        raise ValueError(f"mask shape {shape!r} needs 3 radii")
    if shape == "ellipsoid":
        r = np.sqrt((x ** 2).sum(-1))
        e = np.sqrt(((x / radius) ** 2).sum(-1))
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(e > 0, r * (1 - 1 / e), -radius.min())      # distance outside, along the ray
        return _falloff(d, soft)
    m = np.ones((box,) * 3, np.float32)
    for i in range(3):
        m *= _falloff(np.abs(x[..., i]) - radius[i], soft)
    return m


def _freq(box: int, apix: float) -> np.ndarray:
    k = np.meshgrid(np.fft.fftfreq(box), np.fft.fftfreq(box), np.fft.rfftfreq(box), indexing="ij")
    return np.sqrt(sum(g ** 2 for g in k)) / apix                    # 1/Å, rfft layout [z, y, x]


def bandpass_filter(box: int, apix: float, lowpass: float | None = None, highpass: float | None = None) -> np.ndarray:
    """Fourier weights (rfft layout) keeping resolutions between highpass and lowpass Å; raised-cosine
    edges EDGE_SHELLS Fourier shells wide, centred on the cutoffs (weight 0.5 at the cutoff)."""
    f = _freq(box, apix)
    w = np.ones_like(f, dtype=np.float32)
    width = EDGE_SHELLS / (box * apix)
    if lowpass:
        w *= _falloff(f - 1.0 / lowpass, width)
    if highpass:
        w *= 1 - _falloff(f - 1.0 / highpass, width)
    return w


def bandpass(vol, apix: float, lowpass: float | None = None, highpass: float | None = None) -> np.ndarray:
    if not lowpass and not highpass:
        return np.asarray(vol, np.float32)
    w = bandpass_filter(vol.shape[0], apix, lowpass, highpass)
    return np.fft.irfftn(np.fft.rfftn(vol) * w, s=vol.shape, axes=(0, 1, 2)).astype(np.float32)


def fit_box(vol, box: int) -> np.ndarray:
    """Pad with zeros or crop a cube so its centre voxel n//2 lands on box//2."""
    vol = np.asarray(vol, np.float32)
    n = vol.shape[0]
    out = np.zeros((box,) * 3, np.float32)
    off = box // 2 - n // 2
    src = slice(max(0, -off), min(n, box - off))
    dst = slice(max(0, off), max(0, off) + (src.stop - src.start))
    out[dst, dst, dst] = vol[src, src, src]
    return out


def masked_normalize(vol, mask) -> np.ndarray:
    """Mean 0 / std 1 weighted by ``mask``, then multiplied by the mask."""
    w = mask.astype(np.float64)
    sw = w.sum()
    mean = (w * vol).sum() / sw
    std = np.sqrt((w * (vol - mean) ** 2).sum() / sw)
    return ((vol - mean) / (std if std > 0 else 1) * mask).astype(np.float32)


def masked_cc(a, b, mask) -> float:
    """Pearson correlation of a and b weighted by mask."""
    w = mask.astype(np.float64)
    sw = w.sum()
    a = a - (w * a).sum() / sw
    b = b - (w * b).sum() / sw
    den = np.sqrt((w * a * a).sum() * (w * b * b).sum())
    return float((w * a * b).sum() / den) if den > 0 else float("nan")


def prepare_reference(vol, apix: float, mask, lowpass=None, highpass=None, invert: bool = False) -> np.ndarray:
    """Contrast flip (optional), band-pass, normalise inside the mask, apply the mask."""
    v = -np.asarray(vol, np.float32) if invert else np.asarray(vol, np.float32)
    return masked_normalize(bandpass(v, apix, lowpass, highpass), mask)


def load_volumes(path, box: int, apix: float) -> list[np.ndarray]:
    """Volumes from an MRC (one cube, or a stack of cubes along z), fitted to ``box``.
    The voxel size must match ``apix`` (resampling is not implemented)."""
    h = read_header(path)
    if abs(h["apix"] - apix) > 1e-3 * apix:
        raise ValueError(f"{path}: voxel size {h['apix']:g} Å differs from the project's {apix:g} Å")
    nz, ny, nx = h["shape"]
    if ny != nx or nz % nx:
        raise ValueError(f"{path}: shape {h['shape']} is neither a cube nor a stack of cubes")
    vol, _ = read_mrc(path, mmap=False)
    return [fit_box(v, box) for v in vol.reshape(nz // nx, nx, nx, nx)]


def resolve_file(name, root) -> Path:
    """A path as written (absolute or relative to cwd), else relative to the project ``root``."""
    for c in (Path(name), Path(root) / name):
        if c.is_file():
            return c
    raise FileNotFoundError(f"{name}: not found (relative to {Path.cwd()} or {root})")
