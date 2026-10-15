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

from .average import AverageResult, extraction_sphere, half_sets, sphere_mask
from .io.mrc import read_mrc
from .wedge import TiltInfo, electron_wavelength, particle_defocus


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


def extract_batch(vol, centres: np.ndarray, mats: np.ndarray, box: int, device, grid=None, full_box=True):
    """Batched reference-frame extraction. Returns sub (B,box,box,box), inside (B,box,box,box) on device.

    full_box=False cuts only the cube needed for the extraction sphere (faster; corners then read 0).
    """
    size = int(math.ceil(box * math.sqrt(3))) + 3 if full_box else box + 3
    cubes, lo = crop_batch(vol, centres, size)
    return sample_cubes(cubes, lo, vol.shape, centres, mats, box, device, grid)


def sample_cubes(cubes, lo, tomo_shape, centres, mats, box: int, device, grid=None):
    """Sample (B, S, S, S) cubes whose first voxels are tomogram voxels ``lo`` (B,3 xyz); mirrors
    average.extract / extract_crop. Points outside a cube or outside the tomogram get weight 0."""
    grid = box_grid_t(box, device) if grid is None else grid
    size = cubes.shape[-1]
    M = torch.as_tensor(np.asarray(mats, np.float32), device=device)            # (B,3,3)
    c = torch.as_tensor(np.asarray(centres, np.float32), device=device)          # (B,3)
    pts = torch.einsum("zyxi,bij->bzyxj", grid, M) + c[:, None, None, None, :]   # r @ M + c  (tomo voxels)
    shape = torch.as_tensor(tomo_shape[::-1], dtype=torch.float32, device=device)
    p0 = torch.floor(pts)
    local = pts - torch.as_tensor(np.asarray(lo, np.float32), device=device)[:, None, None, None, :]
    l0 = torch.floor(local)
    inside = (((p0 >= 0) & (p0 + 1 < shape)).all(-1) & ((l0 >= 0) & (l0 + 1 < size)).all(-1)).float()
    norm = local * (2.0 / (size - 1)) - 1.0                                      # align_corners=True
    inp = torch.as_tensor(cubes, device=device).float()[:, None]
    sub = F.grid_sample(inp, norm, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]
    return sub * inside, inside


def normalize_batch(sub, inside, norm_mask=None):
    """mean 0 / std 1 over inside * norm_mask voxels; applied to all inside voxels (= average.prepare_subvolume)."""
    m = inside if norm_mask is None else inside * norm_mask
    n = m.sum((1, 2, 3)).clamp_min(1)
    mean = (sub * m).sum((1, 2, 3)) / n
    var = (((sub - mean[:, None, None, None]) * m) ** 2).sum((1, 2, 3)) / n
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
                        grid=None, device=None, defocus_A=None, ctf_mode=None):
    """(B, box, box, box//2+1) weights; same model as wedge.wedge_weights.

    defocus_A: (B, n_tilts) per-particle defocus (wedge.particle_defocus), needed for ctf_mode.
    """
    kx, ky, kz = grid if grid is not None else rfft_grid_t(box, device)
    dev = kx.device
    m = torch.as_tensor(np.asarray(mats, np.float32), device=dev)  # (B,3,3)
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
    w = torch.where(on_axis, torch.ones_like(w), w)
    if dose_weight or ctf_mode:
        if apix is None:
            raise ValueError("dose / CTF weighting needs apix")
        order = np.argsort(tilt.sign * tilt.angles)
        b_np = (tilt.sign * tilt.angles)[order]
        b_t = torch.as_tensor(b_np, dtype=torch.float32, device=dev)
        j = torch.searchsorted(b_t, beta.contiguous()).clamp(1, len(b_t) - 1)
        nearest = torch.where((beta - b_t[j - 1]).abs() < (beta - b_t[j]).abs(), j - 1, j)
        nearest = torch.where(on_axis, torch.full_like(nearest, int(np.argmin(np.abs(b_np)))), nearest)
        kmag = (kt ** 2).sum(-1).sqrt() / apix
    if dose_weight:
        if tilt.dose is None:
            raise ValueError("dose weighting needs tilt.dose")
        dose = torch.as_tensor(tilt.dose[order], dtype=torch.float32, device=dev)
        ne = 0.245 * kmag.clamp_min(1e-4) ** -1.665 + 2.81  # == wedge.critical_exposure
        w = w * torch.exp(-dose[nearest] / (2 * ne))
    if ctf_mode:
        if tilt.ctf is None or defocus_A is None:
            raise ValueError("CTF weighting needs tilt.ctf and defocus_A")
        df = torch.as_tensor(np.asarray(defocus_A, np.float32)[:, order], device=dev)   # (B, T)
        dfv = torch.gather(df, 1, nearest.reshape(len(df), -1)).reshape(nearest.shape)
        ctf = tilt.ctf
        lam = electron_wavelength(ctf.voltage_kv)
        k2 = kmag ** 2
        chi = math.pi * lam * dfv * k2 - 0.5 * math.pi * ctf.cs_mm * 1e7 * lam ** 3 * k2 ** 2 - ctf.phase_shift_rad
        a = ctf.amplitude
        c = math.sqrt(1 - a * a) * torch.sin(chi) + a * torch.cos(chi)   # = -ctf_values (positive at k->0)
        if ctf_mode == "flip":
            c = torch.where(c < 0, -torch.ones_like(c), torch.ones_like(c))
        elif ctf_mode == "amplitude":
            c = c.abs()
        elif ctf_mode != "wiener":
            raise ValueError(f"unknown ctf_mode {ctf_mode!r}")
        w = w * c
    return w


# ------------------------------------------------------------------- average
@torch.no_grad()
def average_particles_torch(ps, tomograms: dict | None, box: int, rotate: bool = True, normalize: bool = True,
                            min_inside: float = 0.9, progress=None, tilts: dict | None = None,
                            dose_weight: bool = False, wiener_eps: float = 0.05,
                            device=None, batch: int = 32, ctf_mode: str | None = None,
                            height_sign: float = 1.0, jobs=None, norm_radius: float | None = None,
                            stack=None, per_tomo: bool = False, invert: bool = False) -> AverageResult:
    """Same semantics as average.average_particles, on the GPU (also from a ``stack``)."""
    device = torch.device(device) if device is not None else default_device()
    grid = box_grid_t(box, device)
    kgrid = rfft_grid_t(box, device) if tilts is not None else None
    sphere = torch.as_tensor(extraction_sphere(box), device=device)
    nmask = sphere if norm_radius is None else torch.as_tensor(sphere_mask(box, norm_radius), device=device)
    n_sphere = float(sphere.sum())
    fshape = (box, box, box // 2 + 1) if tilts is not None else (box,) * 3
    ctype = torch.complex128 if tilts is not None else torch.float64
    acc = {h: torch.zeros(fshape, dtype=ctype, device=device) for h in (1, 2)}
    wacc = {h: torch.zeros(fshape, dtype=torch.float64, device=device) for h in (1, 2)}
    nh = {1: 0, 2: 0}
    by_tomo = {} if per_tomo else None
    halves_all = half_sets(ps)
    mats_all = ps.matrices if rotate else np.broadcast_to(np.eye(3), (len(ps), 3, 3))
    loc = stack.locate(ps) if stack is not None else None
    data = stack.data("r") if stack is not None else None
    used = skipped = 0
    apix_out = None
    for tomo, idx in ps.df.groupby("rlnTomoName", sort=False).groups.items():
        if loc is None:
            if tomo not in tomograms:
                raise KeyError(f"no tomogram given for rlnTomoName={tomo!r}")
            vol, apix = read_mrc(tomograms[tomo])
            tshape = vol.shape
        else:
            apix, tshape = stack.apix, loc.tomo_shape[tomo]
        apix_out = apix_out or apix
        idx = np.asarray(idx)
        centres = ps.centers_px[idx] * (ps.ts_apix / apix)
        tomo_centre = (np.array(tshape[::-1], float) - 1) / 2
        t_acc = torch.zeros(fshape, dtype=ctype, device=device) if per_tomo else None
        t_w = torch.zeros(fshape, dtype=torch.float64, device=device) if per_tomo else None
        t_n = 0
        for s in range(0, len(idx), batch):
            bi = idx[s:s + batch]
            cb = centres[s:s + batch]
            if loc is None:
                sub, ins = extract_batch(vol, cb, mats_all[bi], box, device, grid, full_box=False)
            else:
                cubes = np.asarray(data[np.sort(loc.rows[bi])])[np.argsort(np.argsort(loc.rows[bi]))]
                sub, ins = sample_cubes(cubes, loc.lo[bi], tshape, cb, mats_all[bi], box, device, grid)
            ins = ins * sphere
            sub = sub * sphere
            if invert:
                sub = -sub
            ok = (ins.sum((1, 2, 3)) / n_sphere >= min_inside).cpu().numpy()
            skipped += int((~ok).sum())
            if not ok.any():
                continue
            sel = torch.as_tensor(np.nonzero(ok)[0], device=device)
            cen = cb[ok]
            sub, ins, bi = sub[sel], ins[sel], bi[ok]
            sub = normalize_batch(sub, ins, nmask) if normalize else sub * ins
            used += len(bi)
            halves = torch.as_tensor(halves_all[bi], device=device)
            if tilts is not None:
                dfc = (np.stack([particle_defocus(tilts[tomo], (c - tomo_centre) * apix, height_sign) for c in cen])
                       if ctf_mode else None)
                w = wedge_weights_batch(box, tilts[tomo], mats_all[bi], apix, dose_weight, grid=kgrid,
                                        defocus_A=dfc, ctf_mode=ctf_mode)
                vals, wts = torch.fft.rfftn(sub, dim=(1, 2, 3)) * w, w * w
            else:
                vals, wts = sub, ins
            for h in (1, 2):
                m = halves == h
                if not m.any():
                    continue
                nh[h] += int(m.sum())
                acc[h] += vals[m].sum(0).to(ctype)
                wacc[h] += wts[m].sum(0).double()
            if per_tomo:
                t_acc += vals.sum(0).to(ctype)
                t_w += wts.sum(0).double()
                t_n += len(bi)
            if progress:
                progress(used + skipped, len(ps))
        if per_tomo:
            by_tomo[tomo] = (t_acc.cpu().numpy(), t_w.cpu().numpy(), t_n)

    if tilts is not None:
        def fin(Fa, Wa, n):
            return torch.fft.irfftn(Fa / Wa.clamp_min(wiener_eps * max(n, 1)), s=(box,) * 3).float().cpu().numpy()
    else:
        def fin(Sa, Wa, n):
            return (Sa / Wa.clamp_min(1e-6)).float().cpu().numpy()
    avg = fin(acc[1] + acc[2], wacc[1] + wacc[2], used)
    halves = (fin(acc[1], wacc[1], nh[1]), fin(acc[2], wacc[2], nh[2]))
    return AverageResult(avg, halves, used, skipped, apix_out, by_tomo, tilts is not None, wiener_eps)
