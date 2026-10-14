"""PyTorch (CUDA) backend for extraction and averaging.

Mirrors the numpy reference in ``average.py`` / ``wedge.py`` operation for
operation, so results agree to float32 precision (checked by
``scripts/check_torch.py``). Particles are processed in batches: a cube of
side ceil(box*sqrt(3))+3 around each particle is cut from the memmapped
tomogram on the CPU, moved to the GPU, and resampled with ``grid_sample``
(trilinear, align_corners=True == numpy voxel-centre convention).

Install for CUDA 12.x drivers (12.2 here):
    pip install torch --index-url https://download.pytorch.org/whl/cu121
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .average import AverageResult
from .io.mrc import read_mrc
from .wedge import TiltInfo


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ geometry
def box_grid_t(box: int, device) -> torch.Tensor:
    r = torch.arange(box, dtype=torch.float32, device=device) - box // 2
    z, y, x = torch.meshgrid(r, r, r, indexing="ij")
    return torch.stack([x, y, z], -1)  # (box,box,box,3) xyz


def crop_batch(vol, centres: np.ndarray, size: int):
    """CPU: cut (B, size, size, size) cubes around centres (xyz); zero padded. Returns cubes, lo (B,3 xyz)."""
    shape = np.array(vol.shape[::-1])
    lo = np.round(centres).astype(int) - size // 2
    out = np.zeros((len(centres), size, size, size), np.float32)
    for b, l in enumerate(lo):
        a = np.clip(l, 0, shape)
        e = np.clip(l + size, 0, shape)
        if np.any(e <= a):
            continue
        d0 = a - l
        d1 = d0 + (e - a)
        out[b, d0[2]:d1[2], d0[1]:d1[1], d0[0]:d1[0]] = vol[a[2]:e[2], a[1]:e[1], a[0]:e[0]]
    return out, lo


def extract_batch(vol, centres: np.ndarray, mats: np.ndarray, box: int, device, grid=None):
    """Batched reference-frame extraction. Returns sub (B,box,box,box), inside (B,box,box,box) on device."""
    size = int(math.ceil(box * math.sqrt(3))) + 3
    cubes, lo = crop_batch(vol, centres, size)
    grid = box_grid_t(box, device) if grid is None else grid
    M = torch.as_tensor(np.asarray(mats, np.float32), device=device)            # (B,3,3)
    c = torch.as_tensor(np.asarray(centres, np.float32), device=device)          # (B,3)
    pts = torch.einsum("zyxi,bij->bzyxj", grid, M) + c[:, None, None, None, :]   # r @ M + c  (tomo voxels)
    shape = torch.as_tensor(vol.shape[::-1], dtype=torch.float32, device=device)
    p0 = torch.floor(pts)
    inside = ((p0 >= 0) & (p0 + 1 < shape)).all(-1).float()
    local = pts - torch.as_tensor(lo, dtype=torch.float32, device=device)[:, None, None, None, :]
    norm = local * (2.0 / (size - 1)) - 1.0                                      # align_corners=True
    inp = torch.as_tensor(cubes, device=device)[:, None]
    sub = F.grid_sample(inp, norm, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]
    return sub * inside, inside


def normalize_batch(sub, inside):
    n = inside.sum((1, 2, 3)).clamp_min(1)
    mean = (sub * inside).sum((1, 2, 3)) / n
    var = (((sub - mean[:, None, None, None]) * inside) ** 2).sum((1, 2, 3)) / n
    std = var.sqrt()
    std = torch.where(std > 0, std, torch.ones_like(std))
    return (sub - mean[:, None, None, None]) / std[:, None, None, None] * inside


# --------------------------------------------------------------------- wedge
def rfft_grid_t(box: int, device):
    k = torch.fft.fftfreq(box, device=device)
    kx = torch.fft.rfftfreq(box, device=device)
    kz, ky, kxx = torch.meshgrid(k, k, kx, indexing="ij")
    return kxx, ky, kz


def wedge_weights_batch(box, tilt: TiltInfo, mats, apix=None, dose_weight=False, soft_deg=2.0,
                        grid=None, device=None):
    """(B, box, box, box//2+1) weights; same model as wedge.wedge_weights."""
    kx, ky, kz = grid if grid is not None else rfft_grid_t(box, device)
    m = torch.as_tensor(np.asarray(mats, np.float32), device=kx.device)  # (B,3,3)
    k = torch.stack([kx, ky, kz], -1)                                   # (...,3)
    kt = torch.einsum("zyxj,bji->bzyxi", k, m)                          # M.T @ k
    tx, ty, tz = kt.unbind(-1)
    flip = tx < 0
    tx = torch.where(flip, -tx, tx)
    tz = torch.where(flip, -tz, tz)
    beta = torch.rad2deg(torch.atan2(tz, tx))
    on_axis = (tx.abs() < 1e-6) & (tz.abs() < 1e-6)
    lo, hi = tilt.range
    if soft_deg > 0:
        d = torch.minimum(beta - lo, hi - beta)
        w = torch.clamp(0.5 + d / soft_deg, 0, 1)
        w = 0.5 - 0.5 * torch.cos(math.pi * w)
    else:
        w = ((beta >= lo) & (beta <= hi)).float()
    if dose_weight:
        if tilt.dose is None or apix is None:
            raise ValueError("dose weighting needs tilt.dose and apix")
        order = np.argsort(tilt.sign * tilt.angles)
        b_t = torch.as_tensor((tilt.sign * tilt.angles)[order], dtype=torch.float32, device=kx.device)
        dose = torch.as_tensor(tilt.dose[order], dtype=torch.float32, device=kx.device)
        j = torch.searchsorted(b_t, beta.contiguous()).clamp(1, len(b_t) - 1)
        nearest = torch.where((beta - b_t[j - 1]).abs() < (beta - b_t[j]).abs(), j - 1, j)
        kmag = (kt ** 2).sum(-1).sqrt() / apix
        ne = 0.245 * kmag.clamp_min(1e-4) ** -1.665 + 2.81  # == wedge.critical_exposure
        w = w * torch.exp(-dose[nearest] / (2 * ne))
    return torch.where(on_axis, torch.ones_like(w), w)


# ------------------------------------------------------------------- average
@torch.no_grad()
def average_particles_torch(ps, tomograms: dict, box: int, rotate: bool = True, normalize: bool = True,
                            min_inside: float = 0.9, progress=None, tilts: dict | None = None,
                            dose_weight: bool = False, wiener_eps: float = 0.05,
                            device=None, batch: int = 32) -> AverageResult:
    """Same semantics as average.average_particles, on the GPU."""
    device = torch.device(device) if device is not None else default_device()
    grid = box_grid_t(box, device)
    kgrid = rfft_grid_t(box, device) if tilts is not None else None
    fshape = (box, box, box // 2 + 1)
    if tilts is not None:
        acc = {h: torch.zeros(fshape, dtype=torch.complex128, device=device) for h in (1, 2)}
        wacc = {h: torch.zeros(fshape, dtype=torch.float64, device=device) for h in (1, 2)}
    else:
        acc = {h: torch.zeros((box,) * 3, dtype=torch.float64, device=device) for h in (1, 2)}
        wacc = {h: torch.zeros((box,) * 3, dtype=torch.float64, device=device) for h in (1, 2)}
    nh = {1: 0, 2: 0}
    subset = ps.df["rlnRandomSubset"].to_numpy(int) if "rlnRandomSubset" in ps.df else None
    mats_all = ps.matrices if rotate else np.broadcast_to(np.eye(3), (len(ps), 3, 3))
    used = skipped = 0
    apix_out = None
    for tomo, idx in ps.df.groupby("rlnTomoName", sort=False).groups.items():
        if tomo not in tomograms:
            raise KeyError(f"no tomogram given for rlnTomoName={tomo!r}")
        vol, apix = read_mrc(tomograms[tomo])
        apix_out = apix_out or apix
        idx = np.asarray(idx)
        centres = ps.centers_px[idx] * (ps.ts_apix / apix)
        for s in range(0, len(idx), batch):
            bi = idx[s:s + batch]
            sub, ins = extract_batch(vol, centres[s:s + batch], mats_all[bi], box, device, grid)
            ok = (ins.mean((1, 2, 3)) >= min_inside).cpu().numpy()
            skipped += int((~ok).sum())
            if not ok.any():
                continue
            sel = torch.as_tensor(np.nonzero(ok)[0], device=device)
            sub, ins, bi = sub[sel], ins[sel], bi[ok]
            if normalize:
                sub = normalize_batch(sub, ins)
            halves = []
            for i in bi:
                h = int(subset[i]) if subset is not None and subset[i] in (1, 2) else (1 if used % 2 == 0 else 2)
                halves.append(h)
                used += 1
            halves = torch.as_tensor(halves, device=device)
            if tilts is not None:
                w = wedge_weights_batch(box, tilts[tomo], mats_all[bi], apix, dose_weight, grid=kgrid)
                Fs = torch.fft.rfftn(sub, dim=(1, 2, 3)) * w
                w2 = w * w
            for h in (1, 2):
                m = halves == h
                if not m.any():
                    continue
                nh[h] += int(m.sum())
                if tilts is not None:
                    acc[h] += Fs[m].sum(0).to(torch.complex128)
                    wacc[h] += w2[m].sum(0).double()
                else:
                    acc[h] += sub[m].sum(0).double()
                    wacc[h] += ins[m].sum(0).double()
            if progress:
                progress(used + skipped, len(ps))

    if tilts is not None:
        def fin(Fa, Wa, n):
            return torch.fft.irfftn(Fa / Wa.clamp_min(wiener_eps * max(n, 1)), s=(box,) * 3).float().cpu().numpy()
    else:
        def fin(Sa, Wa, n):
            return (Sa / Wa.clamp_min(1e-6)).float().cpu().numpy()
    avg = fin(acc[1] + acc[2], wacc[1] + wacc[2], used)
    halves = (fin(acc[1], wacc[1], nh[1]), fin(acc[2], wacc[2], nh[2]))
    return AverageResult(avg, halves, used, skipped, apix_out)
