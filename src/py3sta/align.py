"""Map-to-map alignment (first part of the alignment engine, PLAN.md §3).

Finds the pose (M, t) of the *fixed* map's structure inside the *moving* map, in py3sta's particle
convention: the aligned map is ``moving(c + M.T @ (r + t))`` (``average.extract`` with centre
``c + M.T t``), with c the box centre (voxel box//2), M = euler_to_matrix(rot, tilt, psi) and t the
shift in the fixed (reference) frame. Score = mask-weighted Pearson correlation with the fixed map,
evaluated for all shifts |t_i| <= max_shift at once by FFT.

Search: uniform SO(3) grid on binned, low-passed maps -> local grid around the best candidates ->
local grid at full sampling (integer shifts + parabolic sub-voxel estimate) -> optional polish: Nelder-Mead
over rotation vector *and* sub-voxel shift together, scoring the map extracted at exactly that pose,
accepted only if it improves (after RealSpaceTM ``--polish``, which optimises the rotation against the
best integer-shift score; that objective is biased when the true shift is fractional, because a small
rotation can mimic a half-voxel shift). Optionally the
mirrored moving map is searched too (hand check). The lowest correlation over the search is kept as
a contrast check (an inverted map correlates strongly negatively).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .average import box_grid, extract, fsc, resolution_at
from .geometry import euler_to_matrix, matrix_to_euler
from .io.mrc import read_mrc
from .reference import bandpass, fit_box


# ------------------------------------------------------------------ map preparation
def resample(vol, apix_in: float, apix_out: float) -> tuple[np.ndarray, float]:
    """Fourier crop/pad a cube from apix_in to apix_out, keeping voxel n//2 at the centre.

    The input is first zero-padded so that n * apix_in / apix_out is close to an integer, so the
    returned voxel size (second value) equals apix_out to ~1e-4 relative."""
    vol = np.asarray(vol, np.float32)
    n = vol.shape[0]
    r = apix_in / apix_out
    if abs(r - 1) < 1e-6:
        return vol, apix_in
    cands = range(n, n + int(np.ceil(1 / min(r, 1))) + 3)
    n2 = min(cands, key=lambda k: abs(k * r - round(k * r)))
    m = int(round(n2 * r))
    v = np.roll(fit_box(vol, n2), -(n2 // 2), axis=(0, 1, 2))           # centre voxel -> index 0
    F = np.fft.fftshift(np.fft.fftn(v))
    out = np.zeros((m,) * 3, np.complex64)
    k = min(n2, m)
    a, b = n2 // 2 - k // 2, m // 2 - k // 2
    out[b:b + k, b:b + k, b:b + k] = F[a:a + k, a:a + k, a:a + k]
    res = np.fft.ifftn(np.fft.ifftshift(out)).real * (m / n2) ** 3
    return np.roll(res, m // 2, axis=(0, 1, 2)).astype(np.float32), n2 * apix_in / m


def fourier_shift(vol, shift_xyz) -> np.ndarray:
    """Translate a volume by shift_xyz voxels (periodic, no interpolation loss)."""
    n = vol.shape
    k = np.meshgrid(np.fft.fftfreq(n[0]), np.fft.fftfreq(n[1]), np.fft.rfftfreq(n[2]), indexing="ij")
    sx, sy, sz = shift_xyz
    ph = np.exp(-2j * np.pi * (k[2] * sx + k[1] * sy + k[0] * sz))
    return np.fft.irfftn(np.fft.rfftn(vol) * ph, s=n, axes=(0, 1, 2)).astype(np.float32)


def center_of_mass(vol, frac: float = 0.5) -> np.ndarray:
    """Centroid (x, y, z, voxels from the box centre) of the density above frac * max."""
    w = np.where(vol > frac * vol.max(), vol, 0).astype(np.float64)
    if w.sum() <= 0:
        return np.zeros(3)
    g = box_grid(vol.shape[0])
    return (g * w[..., None]).reshape(-1, 3).sum(0) / w.sum()


def mirror(vol) -> np.ndarray:
    """Mirror x about the centre voxel n//2 (changes the hand)."""
    n = vol.shape[2]
    idx = 2 * (n // 2) - np.arange(n)
    ok = (idx >= 0) & (idx < n)
    out = np.zeros_like(vol)
    out[..., ok] = vol[..., idx[ok]]
    return out


def load_map(path, apix: float, box: int, invert: bool = False, center: bool = False) -> np.ndarray:
    """Read an MRC (any axis order), flip contrast, resample to ``apix``, optionally centre the density
    (centre of mass, Fourier shift) and pad/crop to ``box``."""
    vol, a = read_mrc(path, mmap=False)
    vol = -vol if invert else vol
    vol, _ = resample(vol, a, apix)
    if center:
        vol = fourier_shift(vol, -center_of_mass(vol))
    return fit_box(vol, box)


# ------------------------------------------------------------------ rotations
def so3_grid(step_deg: float) -> np.ndarray:
    """Approximately uniform rotation grid (RELION Euler angles, degrees): tilt in steps of ``step``,
    rot in steps of ~step / sin(tilt), psi in steps of ``step``. Returns (N, 3)."""
    out = []
    for tilt in np.arange(0, 180 + 1e-6, step_deg):
        n_rot = max(1, int(round(360 * np.sin(np.deg2rad(tilt)) / step_deg)))
        for rot in np.arange(n_rot) * 360.0 / n_rot:
            for psi in np.arange(0, 360 - 1e-6, step_deg):
                out.append((rot, tilt, psi))
    return np.asarray(out)


def rotvec_to_matrix(v) -> np.ndarray:
    """Rotation vector(s) (radians, (..., 3)) -> matrices (Rodrigues)."""
    v = np.asarray(v, float)
    th = np.linalg.norm(v, axis=-1)[..., None, None]
    k = v / np.where(th[..., 0] > 0, th[..., 0], 1)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def local_grid(M0, range_deg: float, step_deg: float) -> np.ndarray:
    """Matrices dR @ M0 for rotation vectors on a cubic grid within a ball of ``range_deg``."""
    r = np.arange(-range_deg, range_deg + 1e-6, step_deg)
    v = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    v = v[np.linalg.norm(v, axis=1) <= range_deg + 1e-6]
    return rotvec_to_matrix(np.deg2rad(v)) @ np.asarray(M0)


# ------------------------------------------------------------------ scoring (numpy reference)
@dataclass
class Fixed:
    """FFTs of the fixed map and mask for masked correlation over shifts."""
    box: int
    Ff: np.ndarray        # conj(rfft(m * (f - mean_m f)))
    Fm: np.ndarray        # conj(rfft(m))
    W: float              # sum(m)
    VF: float             # sum(m * (f - mean_m f)^2)
    max_shift: int        # voxels

    @classmethod
    def make(cls, fixed, mask, max_shift: int) -> "Fixed":
        m = np.asarray(mask, np.float64)
        W = m.sum()
        mu = (m * fixed).sum() / W
        fd = m * (fixed - mu)
        return cls(fixed.shape[0], np.conj(np.fft.rfftn(fd)), np.conj(np.fft.rfftn(m)), float(W),
                   float((fd * (fixed - mu)).sum()), int(max_shift))


def shift_window(n: int, s: int) -> np.ndarray:
    return np.arange(-s, s + 1) % n


def ncc_map(fx: Fixed, g) -> np.ndarray:
    """Masked Pearson correlation of the fixed map with g shifted by t, for every periodic shift t:
    ncc(t) = sum m (f - mu_f) g(r + t) / sqrt(VF * (S2(t) - S1(t)^2 / W)). Array [tz, ty, tx]."""
    n = fx.box
    G, G2 = np.fft.rfftn(g), np.fft.rfftn(g * g)
    num = np.fft.irfftn(fx.Ff * G, s=(n,) * 3, axes=(0, 1, 2))
    s1 = np.fft.irfftn(fx.Fm * G, s=(n,) * 3, axes=(0, 1, 2))
    s2 = np.fft.irfftn(fx.Fm * G2, s=(n,) * 3, axes=(0, 1, 2))
    var = np.maximum(s2 - s1 * s1 / fx.W, 1e-12 * fx.W)
    return num / np.sqrt(fx.VF * var)


def best_shift(fx: Fixed, ncc) -> tuple[float, np.ndarray, float]:
    """(max ncc, shift xyz voxels with parabolic sub-voxel refinement, min ncc) within the window."""
    n, s = fx.box, fx.max_shift
    w = shift_window(n, s)
    win = ncc[np.ix_(w, w, w)]
    iz, iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
    peak = float(win[iz, iy, ix])
    shift = np.array([ix - s, iy - s, iz - s], float)
    for k, (axis, i) in enumerate(((2, ix), (1, iy), (0, iz))):          # parabola per axis (periodic map)
        idx = [w[iz], w[iy], w[ix]]
        lo, hi = list(idx), list(idx)
        lo[axis], hi[axis] = (idx[axis] - 1) % n, (idx[axis] + 1) % n
        a, c = float(ncc[tuple(lo)]), float(ncc[tuple(hi)])
        den = a - 2 * peak + c
        if den < 0:
            shift[k] += float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return peak, shift, float(win.min())


def score_rotations(fx: Fixed, moving, mats, device=None, batch: int = 64):
    """For each matrix: (best ncc, integer-voxel shift xyz, min ncc). numpy reference, or torch batches."""
    if device is not None:
        return _score_rotations_torch(fx, moving, mats, device, batch)
    n = fx.box
    grid = box_grid(n)
    c = np.full(3, n // 2, float)
    best, shifts, mins = np.empty(len(mats)), np.empty((len(mats), 3)), np.empty(len(mats))
    w = shift_window(n, fx.max_shift)
    for i, M in enumerate(mats):
        g = extract(moving, c, M, n, grid)[0]
        win = ncc_map(fx, g)[np.ix_(w, w, w)]
        j = np.unravel_index(int(np.argmax(win)), win.shape)
        best[i], mins[i] = win[j], win.min()
        shifts[i] = (j[2] - fx.max_shift, j[1] - fx.max_shift, j[0] - fx.max_shift)
    return best, shifts, mins


def _score_rotations_torch(fx: Fixed, moving, mats, device, batch):
    import torch
    import torch.nn.functional as F

    n, s = fx.box, fx.max_shift
    dev = torch.device(device)
    vol = torch.as_tensor(np.asarray(moving, np.float32), device=dev)[None, None]
    r = torch.arange(n, dtype=torch.float32, device=dev) - n // 2
    zz, yy, xx = torch.meshgrid(r, r, r, indexing="ij")
    grid = torch.stack([xx, yy, zz], -1)
    Ff = torch.as_tensor(fx.Ff, device=dev).to(torch.complex64)
    Fm = torch.as_tensor(fx.Fm, device=dev).to(torch.complex64)
    w = torch.as_tensor(shift_window(n, s), device=dev)
    best, shifts, mins = [], [], []
    for b0 in range(0, len(mats), batch):
        M = torch.as_tensor(np.asarray(mats[b0:b0 + batch], np.float32), device=dev)
        B = len(M)
        pts = torch.einsum("zyxi,bij->bzyxj", grid, M) + float(n // 2)       # r @ M + c (moving voxels)
        p0 = torch.floor(pts)
        inside = ((p0 >= 0) & (p0 + 1 < n)).all(-1).float()
        g = F.grid_sample(vol.expand(B, -1, -1, -1, -1), pts * (2.0 / (n - 1)) - 1.0, mode="bilinear",
                          padding_mode="zeros", align_corners=True)[:, 0] * inside
        G, G2 = torch.fft.rfftn(g, dim=(1, 2, 3)), torch.fft.rfftn(g * g, dim=(1, 2, 3))
        num = torch.fft.irfftn(Ff * G, s=(n,) * 3, dim=(1, 2, 3))
        s1 = torch.fft.irfftn(Fm * G, s=(n,) * 3, dim=(1, 2, 3))
        s2 = torch.fft.irfftn(Fm * G2, s=(n,) * 3, dim=(1, 2, 3))
        ncc = num / torch.sqrt(fx.VF * torch.clamp(s2 - s1 * s1 / fx.W, min=1e-12 * fx.W))
        win = ncc[:, w][:, :, w][:, :, :, w].reshape(B, -1)
        v, j = win.max(1)
        best.append(v.cpu().numpy())
        mins.append(win.min(1).values.cpu().numpy())
        j = j.cpu().numpy()
        k = 2 * s + 1
        shifts.append(np.stack([j % k - s, (j // k) % k - s, j // (k * k) - s], 1))
    return np.concatenate(best), np.concatenate(shifts).astype(float), np.concatenate(mins)


def score_one(fx: Fixed, moving, M):
    """(ncc, sub-voxel shift xyz, min) for one rotation (numpy)."""
    n = fx.box
    g = extract(moving, np.full(3, n // 2, float), M, n)[0]
    return best_shift(fx, ncc_map(fx, g))


# ------------------------------------------------------------------ search
@dataclass
class AlignResult:
    matrix: np.ndarray            # M (fixed frame <- moving frame), as a particle pose
    euler: np.ndarray             # RELION rot, tilt, psi (deg)
    shift: np.ndarray             # t, voxels at the fixed voxel size, fixed (reference) frame
    cc: float
    mirrored: bool
    cc_other_hand: float | None   # best CC of the other hand (None if not searched)
    min_cc: float                 # lowest CC seen (contrast check)
    n_rotations: int
    seconds: float
    log: list = field(default_factory=list)

    @property
    def inverted_contrast_suspected(self) -> bool:
        return -self.min_cc > self.cc

    def centre_offset(self) -> np.ndarray:
        """Shift of the extraction centre in the moving map, voxels (= M.T t)."""
        return self.matrix.T @ self.shift


def _search_hand(fixed, moving, mask, apix, lowpass, highpass, bin_factor, step, max_shift_A,
                 polish, device, n_candidates, log):
    t0 = time.time()
    box = fixed.shape[0]
    # binned, low-passed copies for the global search
    apix_s = apix * bin_factor
    fs, apix_s = resample(fixed, apix, apix_s)
    ms, _ = resample(moving, apix, apix_s)
    mk, _ = resample(mask, apix, apix_s)
    mk = np.clip(mk, 0, 1)
    fs, ms = bandpass(fs, apix_s, lowpass, highpass), bandpass(ms, apix_s, lowpass, highpass)
    fx_s = Fixed.make(fs, mk, max(1, int(round(max_shift_A / apix_s))))
    eul = so3_grid(step)
    mats = euler_to_matrix(*eul.T)
    sc, _, mn = score_rotations(fx_s, ms, mats, device)
    n_rot = len(mats)
    order = np.argsort(sc)[::-1]
    cands = []                                            # distinct top candidates (> step apart)
    for i in order:
        if all(_angle(mats[i], mats[j]) > step for j in cands):
            cands.append(i)
        if len(cands) == n_candidates:
            break
    log.append(f"global search: {n_rot} rotations at {step:g} deg on {fs.shape[0]}^3 @ {apix_s:.2f} Å, "
               f"best CC {sc[order[0]]:.3f}")
    fine = step / 3
    best = (-2.0, None)
    for i in cands:                                       # local grid around each candidate (binned)
        loc = local_grid(mats[i], step, fine)
        s2, _, m2 = score_rotations(fx_s, ms, loc, device)
        n_rot += len(loc)
        mn = np.concatenate([mn, m2])
        k = int(np.argmax(s2))
        if s2[k] > best[0]:
            best = (float(s2[k]), loc[k])
    # full sampling: local grid, then polish
    fb, mb = bandpass(fixed, apix, lowpass, highpass), bandpass(moving, apix, lowpass, highpass)
    fx = Fixed.make(fb, mask, max(1, int(round(max_shift_A / apix))))
    loc = local_grid(best[1], fine, fine / 2)
    s3, _, m3 = score_rotations(fx, mb, loc, device)
    n_rot += len(loc)
    M = loc[int(np.argmax(s3))]
    _, shift, mn_full = score_one(fx, mb, M)
    cc = pose_cc(fb, mb, mask, M, shift)
    log.append(f"local refinement: CC {cc:.3f} at {fine / 2:.2g} deg on {box}^3 @ {apix:.2f} Å")
    if polish:
        M, shift, cc, n_eval = _polish(fb, mb, mask, M, shift, cc, init_deg=fine / 2, max_shift=fx.max_shift)
        n_rot += n_eval
        log.append(f"polish: CC {cc:.4f} ({n_eval} evaluations)")
    min_cc = float(min(mn.min(), m3.min(), mn_full))
    return M, shift, cc, min_cc, n_rot, time.time() - t0


def _angle(A, B) -> float:
    c = (np.trace(np.asarray(A) @ np.asarray(B).T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def pose_cc(fixed, moving, mask, M, shift) -> float:
    """Masked CC of the fixed map with the moving map extracted at pose (M, shift) -- no shift rounding."""
    from .reference import masked_cc

    n = fixed.shape[0]
    g = extract(moving, np.full(3, n // 2, float) + np.asarray(M).T @ np.asarray(shift, float), M, n)[0]
    return masked_cc(fixed, g, mask)


def _polish(fixed, moving, mask, M0, shift0, cc0, init_deg=2.0, tol_deg=0.02, max_iter=400, max_shift=None):
    """Nelder-Mead over (rotation vector around M0, sub-voxel shift), objective = pose_cc; explicit
    initial simplex (the default collapses at 0); shifts beyond ``max_shift`` voxels (per axis) are
    rejected; accepted only if it improves."""
    from scipy.optimize import minimize

    n_eval = 0
    shift0 = np.asarray(shift0, float)

    def neg(x):
        nonlocal n_eval
        n_eval += 1
        if max_shift is not None and np.abs(shift0 + x[3:]).max() > max_shift:
            return 2.0                                       # outside the search range
        return -pose_cc(fixed, moving, mask, rotvec_to_matrix(x[:3]) @ M0, shift0 + x[3:])

    simplex = np.vstack([np.zeros(6), np.diag([np.deg2rad(init_deg)] * 3 + [0.5] * 3)])
    res = minimize(neg, np.zeros(6), method="Nelder-Mead",
                   options=dict(initial_simplex=simplex, xatol=np.deg2rad(tol_deg), fatol=1e-7, maxiter=max_iter))
    if -res.fun > cc0:
        return rotvec_to_matrix(res.x[:3]) @ M0, shift0 + res.x[3:], float(-res.fun), n_eval
    return M0, shift0, cc0, n_eval


def align_maps(fixed, moving, apix: float, mask, lowpass: float | None = 30.0, highpass: float | None = None,
               bin_factor: int = 2, step: float = 10.0, max_shift_A: float = 30.0, polish: bool = True,
               try_mirror: bool = True, device=None, n_candidates: int = 3) -> AlignResult:
    """Align ``moving`` onto ``fixed`` (same box and voxel size; see load_map). ``mask`` weights the
    fixed map. Searches both hands if ``try_mirror`` and keeps the better one."""
    if fixed.shape != moving.shape or fixed.shape != mask.shape:
        raise ValueError(f"fixed {fixed.shape}, moving {moving.shape} and mask {mask.shape} must have one box")
    if lowpass and lowpass < 2 * apix * bin_factor:
        raise ValueError(f"lowpass {lowpass:g} Å is beyond Nyquist of the binned search ({2 * apix * bin_factor:g} Å); "
                         "lower --bin or raise --lowpass")
    log = []
    hands = [(False, moving)] + ([(True, mirror(moving))] if try_mirror else [])
    results = []
    for is_mirror, mv in hands:
        hl = []
        out = _search_hand(fixed, mv, mask, apix, lowpass, highpass, bin_factor, step, max_shift_A, polish,
                           device, n_candidates, hl)
        log += [("mirror: " if is_mirror else "direct: ") + m for m in hl]
        results.append((is_mirror, *out))
    results.sort(key=lambda r: r[3], reverse=True)
    is_mirror, M, shift, cc, min_cc, n_rot, sec = results[0]
    other = results[1][3] if len(results) > 1 else None
    return AlignResult(M, matrix_to_euler(M), shift, cc, is_mirror, other,
                       min(r[4] for r in results), sum(r[5] for r in results), sum(r[6] for r in results), log)


def apply_alignment(moving, res: AlignResult) -> np.ndarray:
    """The moving map in the fixed frame (one trilinear interpolation)."""
    mv = mirror(moving) if res.mirrored else moving
    n = mv.shape[0]
    return extract(mv, np.full(3, n // 2, float) + res.centre_offset(), res.matrix, n)[0]


def map_fsc(a, b, apix: float, mask) -> tuple[float, float, np.ndarray, np.ndarray]:
    """FSC of two masked maps: (resolution at 0.5, at 0.143, resolutions, fsc)."""
    r, f = fsc(a * mask, b * mask, apix)
    return resolution_at(r, f, 0.5), resolution_at(r, f, 0.143), r, f


# ------------------------------------------------------------------ files
def align_map_files(fixed_path, moving_paths, out_dir, mask_file=None, mask_radius: float | None = None,
                    mask_soft: float | None = None, invert: bool = False, center: bool = False, device=None,
                    log=None, **kw) -> "pd.DataFrame":
    """Align each moving MRC onto the fixed MRC (which sets box and voxel size). Writes
    <stem>_aligned.mrc, <stem>_fsc.txt and alignment.star in ``out_dir``; returns the table.

    Default mask: sphere of 40 % of the box, soft edge 10 % (Å)."""
    import os
    from pathlib import Path

    import pandas as pd

    from .io.mrc import write_mrc
    from .io.star import write_star
    from .reference import make_mask

    log = log or (lambda m: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fixed, apix = read_mrc(fixed_path, mmap=False)
    fixed = np.asarray(fixed, np.float32)
    box = fixed.shape[0]
    if fixed.shape != (box,) * 3:
        raise ValueError(f"{fixed_path}: fixed map must be a cube (got {fixed.shape})")
    if mask_file:
        mask = make_mask(box, apix, "file", file=mask_file)
    else:
        mask = make_mask(box, apix, "sphere", [mask_radius or 0.4 * box * apix],
                         soft=mask_soft if mask_soft is not None else 0.1 * box * apix)
    rows = []
    for p in moving_paths:
        mv = load_map(p, apix, box, invert=invert, center=center)
        res = align_maps(fixed, mv, apix, mask, device=device, **kw)
        aligned = apply_alignment(mv, res)
        stem = Path(p).name.split(".")[0]
        write_mrc(out / f"{stem}_aligned.mrc", aligned, apix)
        r05, r0143, r, f = map_fsc(aligned, fixed, apix, mask)
        np.savetxt(out / f"{stem}_fsc.txt", np.column_stack([r, f]), fmt="%.4f", header="resolution_A FSC")
        for m in res.log:
            log(f"{stem}: {m}")
        hand = "mirrored" if res.mirrored else "direct"
        other = f", other hand {res.cc_other_hand:.3f}" if res.cc_other_hand is not None else ""
        log(f"{stem}: {hand} hand, CC {res.cc:.4f}{other}; rot/tilt/psi {np.round(res.euler, 2).tolist()} deg, "
            f"shift {np.round(res.shift * apix, 2).tolist()} Å; FSC 0.5 {r05:.1f} Å, 0.143 {r0143:.1f} Å; "
            f"{res.n_rotations} rotations, {res.seconds:.1f} s")
        if res.inverted_contrast_suspected:
            log(f"{stem}: warning: strongest correlation is negative ({res.min_cc:.3f}): contrast inverted? use --invert")
        rows.append({"py3staMovingMap": os.path.abspath(p), "py3staAlignedMap": str(out / f"{stem}_aligned.mrc"),
                     "rlnAngleRot": res.euler[0], "rlnAngleTilt": res.euler[1], "rlnAnglePsi": res.euler[2],
                     "py3staShiftXAngst": res.shift[0] * apix, "py3staShiftYAngst": res.shift[1] * apix,
                     "py3staShiftZAngst": res.shift[2] * apix, "py3staAlignCC": res.cc,
                     "py3staMirrored": int(res.mirrored),
                     "py3staOtherHandCC": np.nan if res.cc_other_hand is None else res.cc_other_hand,
                     "py3staMinCC": res.min_cc, "py3staInvertedContrast": int(res.inverted_contrast_suspected),
                     "py3staFSC05Angst": r05, "py3staFSC0143Angst": r0143})
    table = pd.DataFrame(rows)
    write_star({"alignment": table}, out / "alignment.star")
    return table
