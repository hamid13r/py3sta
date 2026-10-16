"""MSA: rotated, masked, binned particle stack and PCA (PLAN.md §2 step 4).

Each particle is sampled from its unrotated stack crop at its current pose (one interpolation, every
``bin``-th voxel of the band-passed crop as in mra.py), restricted to the extraction sphere and
normalised inside the mask. Data rows are the voxels inside the (binned, soft) mask, weighted by it.

Modes (``MSASettings.mode``):

* ``wmd`` (default) -- wedge-masked differences (Heumann et al. 2011): d_i = W_i (F_i - A), with A the
  Wiener average of the MSA stack itself, A = sum W_i F_i / max(sum W_i^2, eps n). A particle carries no
  signal where its wedge W_i is empty; subtracting the full average there (plain PCA) leaves -(1 - W_i) A,
  which differs only by wedge orientation and becomes the leading factors ("wedge factors").
* ``plain`` -- x_i - mean, i3 parity; with ``wedge`` each particle is filtered with its own wedge first
  (W_i F_i), the mean is not (wedge factors stay).

PCA is a randomized SVD (Halko, Martinsson & Tropp 2011) of the column-centred rows: torch on the GPU,
numpy as the reference. Factor signs are fixed (largest |component| of each eigenvector positive).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .average import box_grid, extract_crop, extraction_sphere
from .mra import search_lowpass, shrink
from .reference import bandpass_filter
from .stack import particle_centres

MODES = ("wmd", "plain")


@dataclass
class MSASettings:
    mode: str = "wmd"                       # wmd | plain
    wedge: bool = True                      # plain: each particle filtered with its own wedge (wmd always)
    bin: int = 2
    lowpass: float | None = 30.0            # Å
    highpass: float | None = None           # Å
    n_factors: int = 20
    wiener_eps: float = 0.05                # wmd average: sum W^2 floor (as average_particles)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"MSA mode must be one of {', '.join(MODES)} (got {self.mode!r})")

    @property
    def uses_wedge(self) -> bool:
        return self.mode == "wmd" or self.wedge


@dataclass
class MSAStack:
    X: np.ndarray                           # (n, v) float32 rows: masked voxels, weighted by the mask
    voxels: np.ndarray                      # (v,) flat indices into the binned box
    weights: np.ndarray                     # (v,) mask values at those voxels
    box: int                                # binned box
    apix: float                             # binned voxel size
    average: np.ndarray                     # (b, b, b) Wiener average (wmd) or plain mean (plain)

    def volume(self, row) -> np.ndarray:
        """Unmask one row (or an eigenvector) back into the binned box."""
        v = np.zeros(self.box ** 3, np.float32)
        v[self.voxels] = row
        return v.reshape((self.box,) * 3)


@dataclass
class PCAResult:
    scores: np.ndarray                      # (n, k) projections of the centred rows
    vectors: np.ndarray                     # (k, v) eigenvectors (unit rows)
    eigenvalues: np.ndarray                 # (k,) variance along each factor (s^2 / (n - 1))
    var_fraction: np.ndarray                # (k,) fraction of the total variance
    mean: np.ndarray                        # (v,) column mean removed before the SVD
    total_var: float


# ------------------------------------------------------------------ stack
def binned_mask(mask, apix: float, bin: int) -> np.ndarray:
    b = np.asarray(mask).shape[0] // bin
    return (np.clip(shrink(mask, apix, bin), 0, 1) * extraction_sphere(b)).astype(np.float32)


class _Sampler:
    """Band-passed, normalised, binned subvolumes in the reference frame and their wedges (one tomogram
    chunk at a time); numpy reference or torch."""

    def __init__(self, box, apix, s: MSASettings, norm_b, device):
        self.b = box // s.bin
        self.bin, self.apix, self.dev = s.bin, apix, device
        self.lowpass = search_lowpass(s.lowpass, apix, s.bin)
        self.highpass = s.highpass
        self.sphere = extraction_sphere(self.b)
        self.norm = ((norm_b > 0.5) * self.sphere).astype(np.float32)
        if not self.norm.any():
            raise ValueError("normalisation mask has no voxel above 0.5 at the binned sampling")
        self._filt = {}
        if device is not None:
            import torch

            from .backend_torch import box_grid_t, rfft_grid_t

            self.t = torch
            self.grid_t = box_grid_t(self.b, device) * s.bin
            self.kgrid_t = rfft_grid_t(self.b, device)
            self.sphere_t = torch.as_tensor(self.sphere, device=device)
            self.norm_t = torch.as_tensor(self.norm, device=device)
        else:
            from .wedge import rfft_grid

            self.grid = box_grid(self.b) * s.bin
            self.kgrid = rfft_grid(self.b)

    def _filter(self, S):
        if S not in self._filt:
            self._filt[S] = bandpass_filter(S, self.apix, self.lowpass, self.highpass)
        return self._filt[S]

    def sample(self, crops, lo, tomo_shape, centres, mats, tilt):
        """-> (F, W): rfft of the subvolumes (P, b, b, b//2+1) and their wedges (or None), as numpy."""
        if self.dev is not None:
            return self._sample_torch(crops, lo, tomo_shape, centres, mats, tilt)
        from .wedge import wedge_weights

        S = crops.shape[-1]
        w = self._filter(S)
        F, W = [], []
        for crop, l, c, M in zip(crops, lo, centres, mats):
            cf = np.fft.irfftn(np.fft.rfftn(crop) * w, s=crop.shape, axes=(0, 1, 2)).astype(np.float32)
            sub, inside = extract_crop(cf, l, tomo_shape, c, M, self.b, self.grid)
            inside = inside * self.sphere
            m = (inside > 0) & (self.norm > 0)
            v = sub[m]
            sd = v.std() if v.size else 0.0
            sub = np.where(inside > 0, (sub - (v.mean() if v.size else 0.0)) / (sd if sd > 0 else 1), 0)
            F.append(np.fft.rfftn(sub.astype(np.float32)))
            if tilt is not None:
                W.append(wedge_weights(self.b, tilt, M, grid=self.kgrid))
        return np.stack(F), (np.stack(W) if tilt is not None else None)

    def _sample_torch(self, crops, lo, tomo_shape, centres, mats, tilt):
        from .backend_torch import normalize_batch, sample_cubes, wedge_weights_batch

        t = self.t
        S = crops.shape[-1]
        w = t.as_tensor(self._filter(S), device=self.dev)
        c = t.as_tensor(np.asarray(crops, np.float32), device=self.dev)
        cf = t.fft.irfftn(t.fft.rfftn(c, dim=(1, 2, 3)) * w, s=(S,) * 3, dim=(1, 2, 3))
        sub, inside = sample_cubes(cf, lo, tomo_shape, centres, mats, self.b, self.dev, self.grid_t)
        inside = inside * self.sphere_t
        sub = normalize_batch(sub, inside, self.norm_t)
        F = t.fft.rfftn(sub, dim=(1, 2, 3))
        W = None if tilt is None else wedge_weights_batch(self.b, tilt, mats, grid=self.kgrid_t)
        return F.cpu().numpy(), (W.cpu().numpy() if W is not None else None)


class ParticleVolumes:
    """The particles of ``ps`` as MSA data volumes (b^3, reference frame), streamed in tomogram chunks:
    wmd differences W_i (F_i - A) or plain x_i (W_i x_i with ``wedge``). For wmd, the Wiener average A is
    computed first (one extra pass). Iterate for (row indices, (P, b, b, b) float32 volumes); after a full
    iteration ``mean`` is the voxel-wise mean of the data volumes and ``average`` the map they are relative
    to (wmd: A; plain: the mean)."""

    def __init__(self, ps, stack, box: int, apix: float, settings: MSASettings, norm_mask, tilts=None,
                 invert: bool = False, device=None, chunk: int = 64, progress=None):
        s = settings
        if box % s.bin:
            raise ValueError(f"box {box} is not divisible by bin {s.bin}")
        if s.uses_wedge and tilts is None:
            raise ValueError(f"MSA mode {s.mode}{' + wedge' if s.mode == 'plain' else ''} needs tilt information")
        self.s, self.tilts, self.invert, self.chunk, self.progress = s, tilts, invert, chunk, progress
        self.sampler = _Sampler(box, apix, s, binned_mask(norm_mask, apix, s.bin), device)
        self.b = self.sampler.b
        self.apix = apix * s.bin
        self.ps, self.loc = ps, stack.locate(ps)
        self.centres = particle_centres(ps, stack.apix)
        self.mats = ps.matrices
        self.data = stack.data()
        self.tomos = ps.df["rlnTomoName"].to_numpy()
        self.n = len(ps)
        self.total = 2 * self.n if s.mode == "wmd" else self.n
        self.done = 0
        self.A = self._wiener_average() if s.mode == "wmd" else None
        self.mean = None

    def _chunks(self):
        s = self.s
        for tomo in dict.fromkeys(self.tomos):
            tilt = self.tilts.get(tomo) if s.uses_wedge else None
            if s.uses_wedge and tilt is None:
                raise ValueError(f"no tilt information for {tomo}")
            sel = np.nonzero(self.tomos == tomo)[0]
            for c0 in range(0, len(sel), self.chunk):
                ids = sel[c0:c0 + self.chunk]
                crops = np.array(self.data[self.loc.rows[ids]], np.float32)
                if self.invert:
                    crops = -crops
                F, W = self.sampler.sample(crops, self.loc.lo[ids], self.loc.tomo_shape[tomo], self.centres[ids],
                                           self.mats[ids], tilt)
                yield ids, F, W
                self.done += len(ids)
                if self.progress:
                    self.progress(self.done, self.total)

    def _wiener_average(self):
        b = self.b
        num = np.zeros((b, b, b // 2 + 1), np.complex128)
        den = np.zeros((b, b, b // 2 + 1))
        for _, F, W in self._chunks():
            num += (W * F).sum(0)
            den += (W * W).sum(0)
        return (num / np.maximum(den, self.s.wiener_eps * self.n)).astype(np.complex64)

    def __iter__(self):
        shape = (self.b,) * 3
        total = np.zeros(shape, np.float64)
        for ids, F, W in self._chunks():
            if self.s.mode == "wmd":
                G = W * (F - self.A)
            else:
                G = F * W if self.s.wedge else F
            vols = np.fft.irfftn(G, s=shape, axes=(1, 2, 3)).astype(np.float32)
            total += vols.sum(0)
            yield ids, vols
        self.mean = (total / max(self.n, 1)).astype(np.float32)

    @property
    def average(self) -> np.ndarray:
        if self.s.mode == "wmd":
            return np.fft.irfftn(self.A, s=(self.b,) * 3, axes=(0, 1, 2)).astype(np.float32)
        if self.mean is None:
            raise RuntimeError("plain average: iterate over the volumes first")
        return self.mean


def build_stack(ps, stack, mask, apix: float, settings: MSASettings, tilts: dict | None = None,
                invert: bool = False, device=None, chunk: int = 64, progress=None, norm_mask=None) -> MSAStack:
    """MSA rows for every particle of ``ps`` at its current pose (see module doc).

    mask: (box^3) at ``apix``, the MSA mask; norm_mask: where each particle is normalised (default: the
    mask). tilts: {rlnTomoName: TiltInfo}, needed for wmd and for plain + wedge. device: torch device, or
    None for the numpy reference."""
    box = np.asarray(mask).shape[0]
    pv = ParticleVolumes(ps, stack, box, apix, settings, mask if norm_mask is None else norm_mask, tilts, invert,
                         device, chunk, progress)
    mask_b = binned_mask(mask, apix, settings.bin)
    voxels = np.flatnonzero(mask_b.ravel() > 0)
    weights = mask_b.ravel()[voxels]
    X = np.empty((pv.n, len(voxels)), np.float32)
    for ids, vols in pv:
        X[ids] = vols.reshape(len(ids), -1)[:, voxels] * weights
    return MSAStack(X, voxels, weights.astype(np.float32), pv.b, pv.apix, pv.average)


# -------------------------------------------------------------------- PCA
def _signs(V):
    j = np.abs(V).argmax(1)
    return np.where(V[np.arange(len(V)), j] < 0, -1.0, 1.0)


def pca(X, k: int, device=None, n_iter: int = 4, oversample: int = 10, seed: int = 0, omega=None,
        row_chunk: int = 4096) -> PCAResult:
    """Top-``k`` principal components of the rows of X (column-centred). Randomized SVD with ``n_iter``
    power iterations; ``omega`` (v, k + oversample) fixes the random test matrix (parity tests).
    device: torch device or None (numpy). X stays on the host and is streamed in row chunks."""
    X = np.asarray(X, np.float32)
    n, v = X.shape
    k = min(k, n, v)
    p = min(k + oversample, n, v)
    if omega is None:
        omega = np.random.default_rng(seed).standard_normal((v, p)).astype(np.float32)
    mean = X.mean(0, dtype=np.float64).astype(np.float32)
    if device is not None:
        scores, vecs, s, total = _rsvd_torch(X, mean, omega, k, n_iter, device, row_chunk)
    else:
        scores, vecs, s, total = _rsvd_numpy(X, mean, omega, k, n_iter, row_chunk)
    sg = _signs(vecs)
    vecs, scores = vecs * sg[:, None], scores * sg[None]
    ev = s ** 2 / max(n - 1, 1)
    tv = total / max(n - 1, 1)
    return PCAResult(scores.astype(np.float32), vecs.astype(np.float32), ev, ev / tv, mean, float(tv))


def _rsvd_numpy(X, mean, omega, k, n_iter, rc):
    n = len(X)

    def mul(B):                             # (X - mean) @ B
        out = np.empty((n, B.shape[1]), np.float64)
        for r in range(0, n, rc):
            out[r:r + rc] = X[r:r + rc].astype(np.float64) @ B
        return out - (mean.astype(np.float64) @ B)[None]

    def tmul(Y):                            # (X - mean).T @ Y
        out = np.zeros((X.shape[1], Y.shape[1]))
        for r in range(0, n, rc):
            out += X[r:r + rc].T.astype(np.float64) @ Y[r:r + rc]
        return out - np.outer(mean, Y.sum(0))

    Q = np.linalg.qr(mul(omega.astype(np.float64)))[0]
    for _ in range(n_iter):
        Q = np.linalg.qr(tmul(Q))[0]
        Q = np.linalg.qr(mul(Q))[0]
    B = tmul(Q).T                           # (p, v) = Q.T (X - mean)
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    total = sum(float(((X[r:r + rc].astype(np.float64) - mean) ** 2).sum()) for r in range(0, n, rc))
    return (Q @ Ub)[:, :k] * s[:k], Vt[:k], s[:k], total


def _rsvd_torch(X, mean, omega, k, n_iter, device, rc):
    import torch

    t = torch
    n = len(X)
    dt = t.float64
    mu = t.as_tensor(mean, device=device, dtype=dt)
    rows = [t.as_tensor(X[r:r + rc], device=device) for r in range(0, n, rc)] \
        if X.nbytes < 4e9 else None

    def chunk(i, r):
        return (rows[i] if rows is not None else t.as_tensor(X[r:r + rc], device=device)).to(dt)

    def mul(B):
        out = t.cat([chunk(i, r) @ B for i, r in enumerate(range(0, n, rc))])
        return out - (mu @ B)[None]

    def tmul(Y):
        out = t.zeros((X.shape[1], Y.shape[1]), device=device, dtype=dt)
        for i, r in enumerate(range(0, n, rc)):
            out += chunk(i, r).T @ Y[r:r + rc]
        return out - t.outer(mu, Y.sum(0))

    with t.no_grad():
        Q = t.linalg.qr(mul(t.as_tensor(omega, device=device, dtype=dt)))[0]
        for _ in range(n_iter):
            Q = t.linalg.qr(tmul(Q))[0]
            Q = t.linalg.qr(mul(Q))[0]
        B = tmul(Q).T
        Ub, s, Vt = t.linalg.svd(B, full_matrices=False)
        total = sum(float(((chunk(i, r) - mu) ** 2).sum()) for i, r in enumerate(range(0, n, rc)))
        sc = ((Q @ Ub)[:, :k] * s[:k]).cpu().numpy()
        return sc, Vt[:k].cpu().numpy(), s[:k].cpu().numpy(), total


def eigenvolumes(st: MSAStack, res: PCAResult) -> np.ndarray:
    """(k + 1, b, b, b): the average, then each eigenvector unmasked into the binned box (divided by the
    mask weight where it is > 0.05, so a soft edge does not shade the eigenvolumes)."""
    w = np.where(st.weights > 0.05, st.weights, np.inf)
    return np.stack([st.average] + [st.volume(v / w) for v in res.vectors]).astype(np.float32)


# ------------------------------------------------------------ diagnostics
def wedge_features(mats) -> np.ndarray:
    """(n, 12) wedge-orientation features in the reference frame: the second moments (unique entries of
    a a^T) of the tilt axis a = M e_y and of the beam direction z = M e_z. The wedge is symmetric under
    a -> -a and z -> -z, so first moments carry no wedge information."""
    M = np.asarray(mats, float)
    iu = np.triu_indices(3)
    out = []
    for axis in (1, 2):
        a = M[:, :, axis]
        out.append(np.einsum("ni,nj->nij", a, a)[:, iu[0], iu[1]])
    return np.concatenate(out, 1)


def _adjusted(r2: float, n: int, p: int) -> float:
    """sqrt of the adjusted R^2 (chance level ~0 for any n, p): 1 - (1 - R^2) (n - 1) / (n - p - 1)."""
    if n - p - 1 <= 0:
        return float("nan")
    return float(math.sqrt(max(0.0, 1 - (1 - r2) * (n - 1) / (n - p - 1))))


def multiple_r(y, F, adjusted: bool = True) -> float:
    """Multiple correlation of y with the columns of F (least squares with an intercept); ``adjusted``
    corrects for the number of columns (12 random features give R ~ sqrt(12 / n) otherwise)."""
    y = np.asarray(y, float)
    F = np.asarray(F, float)
    A = np.column_stack([np.ones(len(y)), F])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - A @ coef
    ss = ((y - y.mean()) ** 2).sum()
    if ss == 0:
        return 0.0
    r2 = max(0.0, 1 - (r @ r) / ss)
    return _adjusted(r2, len(y), F.shape[1]) if adjusted else float(math.sqrt(r2))


def correlation_ratio(y, groups, adjusted: bool = True) -> float:
    """eta: how much of the variance of y is explained by a categorical variable (one-way ANOVA);
    ``adjusted`` corrects for the number of groups."""
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    ss = ((y - y.mean()) ** 2).sum()
    g = np.unique(groups)
    if ss == 0 or len(g) < 2:
        return 0.0
    between = sum(len(y[groups == v]) * (y[groups == v].mean() - y.mean()) ** 2 for v in g)
    return _adjusted(between / ss, len(y), len(g) - 1) if adjusted else float(math.sqrt(between / ss))


def factor_diagnostics(scores, mats, tomos=None, extra: dict | None = None):
    """Per factor: wedge_r (adjusted multiple correlation with the wedge orientation), tomo_eta (adjusted
    correlation ratio with the tomogram), and |r| with each ``extra`` column (e.g. z height, alignment CC). DataFrame, one row per
    factor (1-based ``factor``)."""
    import pandas as pd

    F = wedge_features(mats)
    rows = []
    for j in range(scores.shape[1]):
        y = scores[:, j]
        row = {"factor": j + 1, "wedge_r": round(multiple_r(y, F), 4)}
        if tomos is not None:
            row["tomo_eta"] = round(correlation_ratio(y, tomos), 4)
        for name, x in (extra or {}).items():
            x = np.asarray(x, float)
            row[f"{name}_r"] = round(abs(float(np.corrcoef(y, x)[0, 1])) if x.std() > 0 and y.std() > 0 else 0.0, 4)
        rows.append(row)
    return pd.DataFrame(rows)
