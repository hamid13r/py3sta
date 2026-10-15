"""Particle alignment against one or more references (PLAN.md §2 step 3, §3 "Alignment engine").

Scoring is done in the reference frame. For a trial pose M' = D M (D: rotation change in the reference
frame, M: current particle matrix) the particle is sampled from its unrotated stack crop at M' (one
interpolation of the raw data) and compared with the reference by masked Pearson correlation over every
shift t at once (FFT; align.ncc_map). The best (D, t) is a pose change in exactly the sense of poses.py:
the new subvolume is the old one at D.T (r + t) -> M'' = D M, c'' = c + M''.T t.

* Missing wedge (``wedge``): the reference AND the sampled particle are filtered with the particle's tilt
  coverage rotated into the reference frame (wedge.wedge_weights(matrix=M')), so only Fourier regions the
  particle sampled are compared ("constrained" correlation). Filtering only the reference made the score
  erratic near the optimum (the particle's actual coverage after interpolation differs from the model).
* Band-pass: applied to each crop once (rotation does not change it), and to the references.
* ``bin``: the particle is sampled every ``bin`` voxels (alias-free: the crop is low-passed to the binned
  Nyquist first) and the references / mask are Fourier-resampled; shifts are reported in full voxels.
* Search: i3-style levels ``[nut, nstep, spin, sstep]`` (cone_grid) around the current pose, coarse to
  fine; each level is centred on the previous best. Every reference is searched separately; the
  reference with the best correlation wins.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .align import Fixed, best_shift, ncc_map, resample, shift_window
from .average import box_grid, extract_crop, extraction_sphere
from .reference import bandpass_filter, fit_box
from .stack import particle_centres


# -------------------------------------------------------------------- grid
def _axis_angle(axis, angle) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


@dataclass
class ConeGrid:
    mats: np.ndarray            # (K, 3, 3) rotation changes D (reference frame); mats[0] = identity
    theta: np.ndarray           # (K,) tilt of the reference z axis, deg
    spin: np.ndarray            # (K,) spin about it, deg
    nut: float
    spin_max: float

    def at_edge(self, k: int) -> bool:
        return bool((self.nut > 0 and self.theta[k] >= self.nut - 1e-6)
                    or (self.spin_max > 0 and abs(self.spin[k]) >= self.spin_max - 1e-6))


def cone_grid(nut: float, nstep: float, spin: float, sstep: float) -> ConeGrid:
    """i3 MRASRCH grid around the current pose: the reference z axis is tilted by theta = 0, nstep, ..
    <= nut in every direction (azimuths ~nstep apart along the cone), and the particle is spun about it
    by -spin .. +spin in steps of sstep. ``0 0 spin sstep`` = rotation about z only; ``0 0 0 0`` = none.
    The new pose's z axis (in the tomogram) is the old one tilted by theta."""
    thetas = [0.0] if nut <= 0 or nstep <= 0 else list(np.arange(0.0, nut + 1e-6, nstep))
    spins = [0.0] if spin <= 0 or sstep <= 0 else list(np.arange(-spin, spin + 1e-6, sstep))
    if 0.0 not in [round(s, 9) for s in spins]:
        spins.append(0.0)
    spins = sorted(spins, key=lambda s: (abs(s), s))
    mats, th, sp = [], [], []
    for t in thetas:
        n_phi = 1 if t == 0 else max(1, int(round(360.0 * math.sin(math.radians(t)) / nstep)))
        for j in range(n_phi):
            phi = 2 * math.pi * j / n_phi
            T = _axis_angle((-math.sin(phi), math.cos(phi), 0.0), math.radians(t)) if t else np.eye(3)
            for s in spins:
                c, si = math.cos(math.radians(s)), math.sin(math.radians(s))
                Rz = np.array([[c, -si, 0], [si, c, 0], [0, 0, 1.0]])
                mats.append(Rz @ T.T)                  # D.T e_z = T e_z: the axis tilted by t
                th.append(t)
                sp.append(s)
    return ConeGrid(np.array(mats), np.array(th), np.array(sp), float(nut), float(spin))


# ---------------------------------------------------------------- settings
@dataclass
class MRASettings:
    levels: list = field(default_factory=lambda: [[12.0, 4.0, 12.0, 4.0], [3.0, 1.0, 3.0, 1.0], [1.0, 0.5, 1.0, 0.5]])
    max_shift: int = 3                      # voxels at full sampling, per tomogram axis
    lowpass: float | None = 30.0            # Å
    highpass: float | None = None           # Å
    bin: int = 1
    wedge: bool = True


@dataclass
class MRAResult:
    R: np.ndarray                           # (n, 3, 3) rotation change (reference frame)
    t: np.ndarray                           # (n, 3) shift, voxels at full sampling (reference frame)
    cc: np.ndarray                          # (n,) best correlation
    best_ref: np.ndarray                    # (n,) 1-based reference index
    cc_ref: np.ndarray                      # (n, n_refs) best correlation per reference
    at_edge: np.ndarray                     # (n,) best pose on the border of the first level / shift limit
    angle: np.ndarray = None                # (n,) rotation angle of R, deg

    def __post_init__(self):
        if self.angle is None:
            tr = np.einsum("nii->n", self.R)
            self.angle = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


@dataclass
class Prepared:
    """References and mask at the search sampling, plus filters."""
    refs: np.ndarray                        # (n_refs, b, b, b) band-passed, at the binned sampling
    mask: np.ndarray                        # (b, b, b) alignment mask * extraction sphere
    box: int                                # binned box
    bin: int
    apix: float                             # full-sampling voxel size
    max_shift: int                          # binned voxels
    lowpass: float | None                   # effective (>= binned Nyquist)
    highpass: float | None


def prepare(refs, mask, apix: float, s: MRASettings) -> Prepared:
    refs = np.asarray(refs, np.float32)
    if refs.ndim == 3:
        refs = refs[None]
    box = refs.shape[-1]
    if box % s.bin:
        raise ValueError(f"box {box} is not divisible by bin {s.bin}")
    b = box // s.bin
    nyq = 2 * apix * s.bin
    lowpass = max(s.lowpass, nyq) if s.lowpass else (nyq if s.bin > 1 else None)

    def shrink(v):
        return fit_box(resample(v, apix, apix * s.bin)[0], b) if s.bin > 1 else np.asarray(v, np.float32)

    w = bandpass_filter(b, apix * s.bin, lowpass, s.highpass)
    out = np.stack([np.fft.irfftn(np.fft.rfftn(shrink(r)) * w, s=(b,) * 3, axes=(0, 1, 2)) for r in refs])
    m = np.clip(shrink(mask), 0, 1) * extraction_sphere(b)
    return Prepared(out.astype(np.float32), m.astype(np.float32), b, s.bin, apix,
                    max(0, int(math.floor(s.max_shift / s.bin))), lowpass, s.highpass)


def filter_crop(crop, pr: Prepared) -> np.ndarray:
    """Band-pass a raw crop (full sampling) to the search band (incl. the binned Nyquist)."""
    crop = np.asarray(crop, np.float32)
    w = bandpass_filter(crop.shape[0], pr.apix, pr.lowpass, pr.highpass)
    return np.fft.irfftn(np.fft.rfftn(crop) * w, s=crop.shape, axes=(0, 1, 2)).astype(np.float32)


# ------------------------------------------------------- numpy reference
def _fixed(pr: Prepared, ref, W=None) -> Fixed:
    f = ref if W is None else np.fft.irfftn(np.fft.rfftn(ref) * W, s=ref.shape, axes=(0, 1, 2))
    return Fixed.make(f, pr.mask, pr.max_shift)


def score_numpy(pr: Prepared, crop_f, lo, tomo_shape, centre, mats, ref_index: int = 0, tilt=None):
    """(cc, integer shift xyz in binned voxels, ncc maps) for each trial matrix M' (numpy reference).
    crop_f: band-passed crop (filter_crop); centre: particle centre, tomogram voxels."""
    from .wedge import rfft_grid, wedge_weights

    grid = box_grid(pr.box) * pr.bin
    sphere = extraction_sphere(pr.box)
    kgrid = rfft_grid(pr.box)
    fx0 = None if tilt is not None else _fixed(pr, pr.refs[ref_index])
    w = shift_window(pr.box, pr.max_shift)
    out_cc, out_t, maps = [], [], []
    for M in mats:
        sub = extract_crop(crop_f, lo, tomo_shape, centre, M, pr.box, grid)[0] * sphere
        if tilt is None:
            fx = fx0
        else:
            W = wedge_weights(pr.box, tilt, M, grid=kgrid)
            fx = _fixed(pr, pr.refs[ref_index], W)
            sub = np.fft.irfftn(np.fft.rfftn(sub) * W, s=sub.shape, axes=(0, 1, 2))
        ncc = ncc_map(fx, sub)
        win = ncc[np.ix_(w, w, w)]
        j = np.unravel_index(int(np.argmax(win)), win.shape)
        out_cc.append(float(win[j]))
        out_t.append((j[2] - pr.max_shift, j[1] - pr.max_shift, j[0] - pr.max_shift))
        maps.append(ncc)
    return np.array(out_cc), np.array(out_t, float), maps


# ---------------------------------------------------------- torch batches
class _TorchScorer:
    def __init__(self, pr: Prepared, device, batch: int):
        import torch

        from .backend_torch import box_grid_t, rfft_grid_t

        self.torch, self.pr, self.dev, self.batch = torch, pr, device, batch
        n = pr.box
        self.grid = box_grid_t(n, device) * pr.bin
        self.kgrid = rfft_grid_t(n, device)
        self.sphere = torch.as_tensor(extraction_sphere(n), device=device)
        self.m = torch.as_tensor(pr.mask, device=device)
        self.Wm = float(pr.mask.sum())
        self.Fm = torch.conj(torch.fft.rfftn(self.m))
        self.Fref = torch.fft.rfftn(torch.as_tensor(pr.refs, device=device), dim=(1, 2, 3))
        self.fixed0 = [self._fixed(self.Fref[r][None]) for r in range(len(pr.refs))]
        self.win = torch.as_tensor(shift_window(n, pr.max_shift), device=device)
        self._crop_filter = None

    def filter_crops(self, crops):
        """Band-pass raw crops (P, S, S, S) on the device (= filter_crop)."""
        t = self.torch
        S = crops.shape[-1]
        if self._crop_filter is None or self._crop_filter.shape[0] != S:
            self._crop_filter = t.as_tensor(bandpass_filter(S, self.pr.apix, self.pr.lowpass, self.pr.highpass),
                                            device=self.dev)
        c = t.as_tensor(crops, device=self.dev)
        return t.fft.irfftn(t.fft.rfftn(c, dim=(1, 2, 3)) * self._crop_filter, s=(S,) * 3, dim=(1, 2, 3))

    def _fixed(self, Fr):
        t = self.torch
        n = self.pr.box
        f = t.fft.irfftn(Fr, s=(n,) * 3, dim=(1, 2, 3))
        mu = (f * self.m).sum((1, 2, 3)) / self.Wm
        fc = f - mu[:, None, None, None]
        fd = self.m * fc
        return t.conj(t.fft.rfftn(fd, dim=(1, 2, 3))), (fd * fc).sum((1, 2, 3))

    def ncc(self, sub, ref_index: int, W=None):
        t = self.torch
        n = self.pr.box
        Ff, VF = self.fixed0[ref_index] if W is None else self._fixed(self.Fref[ref_index][None] * W)
        G = t.fft.rfftn(sub, dim=(1, 2, 3))
        if W is not None:                                   # the particle gets the same coverage filter
            G = G * W
            sub = t.fft.irfftn(G, s=(n,) * 3, dim=(1, 2, 3))
        G2 = t.fft.rfftn(sub * sub, dim=(1, 2, 3))
        num = t.fft.irfftn(Ff * G, s=(n,) * 3, dim=(1, 2, 3))
        s1 = t.fft.irfftn(self.Fm * G, s=(n,) * 3, dim=(1, 2, 3))
        s2 = t.fft.irfftn(self.Fm * G2, s=(n,) * 3, dim=(1, 2, 3))
        var = t.clamp(s2 - s1 * s1 / self.Wm, min=1e-12 * self.Wm)
        return num / t.sqrt(VF[:, None, None, None] * var)

    def score(self, cubes, idx, lo, tomo_shape, centres, mats, ref_index, tilt=None, keep_maps=False):
        """cubes: (P, S, S, S) band-passed crops on device; idx: (B,) cube of each trial."""
        from .backend_torch import sample_cubes, wedge_weights_batch

        t = self.torch
        s = self.pr.max_shift
        k = 2 * s + 1
        ccs, shifts, maps = [], [], []
        for b0 in range(0, len(mats), self.batch):
            sl = slice(b0, b0 + self.batch)
            sub, _ = sample_cubes(cubes[idx[sl]], lo[sl], tomo_shape, centres[sl], mats[sl], self.pr.box,
                                  self.dev, self.grid)
            sub = sub * self.sphere
            W = None if tilt is None else wedge_weights_batch(self.pr.box, tilt, mats[sl], grid=self.kgrid)
            ncc = self.ncc(sub, ref_index, W)
            win = ncc[:, self.win][:, :, self.win][:, :, :, self.win].reshape(len(sub), -1)
            v, j = win.max(1)
            ccs.append(v)
            shifts.append(t.stack([j % k - s, (j // k) % k - s, j // (k * k) - s], 1))
            if keep_maps:
                maps.append(ncc)
        cc = t.cat(ccs).cpu().numpy().astype(float)
        sh = t.cat(shifts).cpu().numpy().astype(float)
        return cc, sh, (t.cat(maps).cpu().numpy() if keep_maps else None)


# ------------------------------------------------------------------ driver
def align_particles(ps, stack, refs, mask, apix: float, settings: MRASettings, tilts: dict | None = None,
                    invert: bool = False, device=None, batch: int = 256, chunk: int = 32,
                    progress=None) -> MRAResult:
    """Align every particle of ``ps`` (from its ``stack`` crop) to each reference; keep the best.

    refs: (n_refs, box, box, box) at the project voxel size ``apix`` (contrast as the particles after
    ``invert``); mask: alignment mask (box^3). tilts: {rlnTomoName: TiltInfo} for the constrained
    correlation (settings.wedge). device: torch device or None (numpy reference, slow).

    After every level the particle is re-centred on the sub-voxel correlation peak, so the next level's
    rotations are compared without a fractional-shift bias (a few degrees of rotation can otherwise mimic
    half a voxel of shift). The total shift is kept within ``max_shift`` voxels per tomogram axis (the
    stack crop allowance)."""
    pr = prepare(refs, mask, apix, settings)
    loc = stack.locate(ps, shift_limit=settings.max_shift)
    grids = [cone_grid(*lv) for lv in settings.levels] or [cone_grid(0, 0, 0, 0)]
    n, nref = len(ps), len(pr.refs)
    centres = particle_centres(ps, stack.apix)
    mats0 = ps.matrices
    data = stack.data()
    scorer = _TorchScorer(pr, device, batch) if device is not None else None
    use_wedge = settings.wedge and tilts is not None
    fx_peak = Fixed(pr.box, None, None, 0.0, 0.0, pr.max_shift)
    lim = float(settings.max_shift)

    best_D = np.zeros((n, nref, 3, 3))
    best_t = np.zeros((n, nref, 3))
    cc_ref = np.full((n, nref), -np.inf)
    edge = np.zeros((n, nref), bool)
    tomos = ps.df["rlnTomoName"].to_numpy()
    done = 0
    for tomo in dict.fromkeys(tomos):
        tilt = tilts.get(tomo) if use_wedge else None
        if use_wedge and tilt is None:
            raise ValueError(f"no tilt information for {tomo}")
        shape = loc.tomo_shape[tomo]
        sel = np.nonzero(tomos == tomo)[0]
        for c0 in range(0, len(sel), chunk):
            ids = sel[c0:c0 + chunk]
            P = len(ids)
            crops = np.array(data[loc.rows[ids]], np.float32)
            if invert:
                crops = -crops
            if scorer:
                cubes = scorer.filter_crops(crops)
            else:
                crops = np.stack([filter_crop(c, pr) for c in crops])
            lo = loc.lo[ids]

            def evaluate(p_of, M, cen, r, keep_maps=False):
                if scorer:
                    return scorer.score(cubes, p_of, lo[p_of], shape, cen, M, r, tilt, keep_maps)
                out = [score_numpy(pr, crops[p], lo[p], shape, cen[j], M[j:j + 1], r, tilt) for j, p in enumerate(p_of)]
                return (np.array([o[0][0] for o in out]), np.array([o[1][0] for o in out]),
                        np.stack([o[2][0] for o in out]) if keep_maps else None)

            def recentre(D, cen, r):
                """Sub-voxel peak at pose D M; returns (peak cc, new centres, shift-limit flags)."""
                M = np.einsum("pij,pjl->pil", D, mats0[ids])
                _, _, maps = evaluate(np.arange(P), M, cen, r, keep_maps=True)
                cc = np.empty(P)
                new = cen.copy()
                hit = np.zeros(P, bool)
                for p in range(P):
                    cc[p], t_sub, _ = best_shift(fx_peak, maps[p])
                    # total displacement, clipped per TOMOGRAM axis: the stack crop allowance is per tomogram
                    # axis (a ±lim cube in the reference frame reaches lim·√3 along a tomogram axis)
                    d = cen[p] + M[p].T @ (t_sub * pr.bin) - centres[ids[p]]
                    hit[p] = lim > 0 and np.any(np.abs(d) >= lim - 1e-3)               # clipped at the limit
                    new[p] = centres[ids[p]] + np.clip(d, -lim, lim)
                return cc, new, hit

            for r in range(nref):
                D = np.repeat(np.eye(3)[None], P, 0)
                cen = centres[ids].copy()
                for li, g in enumerate(grids):
                    K = len(g.mats)
                    trial = np.einsum("kij,pjl->pkil", g.mats, D)                       # G_k D_p
                    M = np.einsum("pkij,pjl->pkil", trial, mats0[ids]).reshape(-1, 3, 3)
                    p_of = np.repeat(np.arange(P), K)
                    cc = evaluate(p_of, M, cen[p_of], r)[0].reshape(P, K)
                    kbest = cc.argmax(1)
                    D = trial[np.arange(P), kbest]
                    if li == 0:
                        edge[ids, r] = [g.at_edge(k) for k in kbest]
                    cc, cen, hit = recentre(D, cen, r)
                cc, cen, hit = recentre(D, cen, r)          # converge the shift at the final rotation
                M = np.einsum("pij,pjl->pil", D, mats0[ids])
                cc_ref[ids, r] = cc
                best_t[ids, r] = np.einsum("pij,pj->pi", M, cen - centres[ids])
                best_D[ids, r] = D
                edge[ids, r] |= hit
            done += P
            if progress:
                progress(done, n)
    k = cc_ref.argmax(1)
    i = np.arange(n)
    return MRAResult(best_D[i, k], best_t[i, k], cc_ref[i, k], k + 1, cc_ref, edge[i, k])
