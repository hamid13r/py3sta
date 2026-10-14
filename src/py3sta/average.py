"""Subvolume extraction and averaging (CPU / numpy reference implementation).

Geometry: for a particle with matrix M (tomogram -> reference, see geometry.py)
and centre c (tomogram voxels, x,y,z), the reference-frame subvolume is

    sub(r) = tomo(c + M.T @ r),   r = voxel offset from the box centre (box//2)

which is RELION's convention (particle = reference actively rotated by A = M.T).
The tomogram is read through a memmap; only a small crop per particle is loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io.mrc import read_mrc
from .io.particles import ParticleSet, tomo_name_from


def box_grid(box: int) -> np.ndarray:
    """(box, box, box, 3) xyz offsets from the centre voxel box//2, array order [z, y, x]."""
    r = np.arange(box, dtype=np.float32) - box // 2
    z, y, x = np.meshgrid(r, r, r, indexing="ij")
    return np.stack([x, y, z], -1)


def _trilinear(vol, pts):
    """Sample vol[z,y,x] at pts (N,3) xyz. Returns (values, inside-mask)."""
    shape = np.array(vol.shape[::-1])  # xyz sizes
    p0 = np.floor(pts).astype(np.int64)
    f = (pts - p0).astype(np.float32)
    inside = np.all((p0 >= 0) & (p0 + 1 < shape), axis=1)
    p0c = np.clip(p0, 0, shape - 2)
    x0, y0, z0 = p0c.T
    fx, fy, fz = f.T
    out = np.zeros(len(pts), np.float32)
    for dz in (0, 1):
        wz = fz if dz else 1 - fz
        for dy in (0, 1):
            wy = fy if dy else 1 - fy
            for dx in (0, 1):
                wx = fx if dx else 1 - fx
                out += wz * wy * wx * vol[z0 + dz, y0 + dy, x0 + dx]
    return np.where(inside, out, 0.0).astype(np.float32), inside


def extract(tomo, centre_xyz, matrix, box: int, grid=None):
    """Return (subvolume[z,y,x], weight[z,y,x]) in the reference frame."""
    grid = box_grid(box) if grid is None else grid
    pts = grid.reshape(-1, 3) @ np.asarray(matrix, np.float32) + np.asarray(centre_xyz, np.float32)
    shape = np.array(tomo.shape[::-1])
    lo = np.clip(np.floor(pts.min(0)).astype(int), 0, shape - 1)
    hi = np.clip(np.ceil(pts.max(0)).astype(int) + 2, 1, shape)
    if np.any(hi - lo < 2):
        z = np.zeros((box,) * 3, np.float32)
        return z, z
    crop = np.asarray(tomo[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]], dtype=np.float32)
    vals, inside = _trilinear(crop, pts - lo)
    return vals.reshape((box,) * 3), inside.reshape((box,) * 3).astype(np.float32)


@dataclass
class AverageResult:
    average: np.ndarray
    halves: tuple | None
    n_used: int
    n_skipped: int
    apix: float


def average_particles(ps: ParticleSet, tomograms: dict, box: int, rotate: bool = True,
                      normalize: bool = True, min_inside: float = 0.9, progress=None,
                      tilts: dict | None = None, dose_weight: bool = False,
                      wiener_eps: float = 0.05) -> AverageResult:
    """Global average of all particles.

    tomograms: {rlnTomoName: path}. Particles whose box is < ``min_inside``
    inside the tomogram are skipped. If rlnRandomSubset exists, half maps are
    also returned.

    tilts: {rlnTomoName: TiltInfo}. If given, the average is wedge-weighted in
    Fourier space:  F = sum(w_i F_i) / max(sum(w_i^2), wiener_eps * n),
    with w_i the particle's tilt coverage rotated into the reference frame.
    """
    if tilts is not None:
        return _average_wedge(ps, tomograms, box, rotate, normalize, min_inside, progress,
                              tilts, dose_weight, wiener_eps)
    grid = box_grid(box)
    shp = (box,) * 3
    sums = {1: np.zeros(shp, np.float64), 2: np.zeros(shp, np.float64)}
    wts = {1: np.zeros(shp, np.float64), 2: np.zeros(shp, np.float64)}
    subset = ps.df["rlnRandomSubset"].to_numpy(int) if "rlnRandomSubset" in ps.df else None
    mats = ps.matrices if rotate else np.broadcast_to(np.eye(3), (len(ps), 3, 3))
    used = skipped = 0
    apix_out = None
    for tomo_name, idx in ps.df.groupby("rlnTomoName", sort=False).groups.items():
        if tomo_name not in tomograms:
            raise KeyError(f"no tomogram given for rlnTomoName={tomo_name!r}; have {list(tomograms)}")
        vol, apix = read_mrc(tomograms[tomo_name])
        apix_out = apix_out or apix
        centres = ps.centers_px[np.asarray(idx)] * (ps.ts_apix / apix)
        for k, i in enumerate(idx):
            sub, w = extract(vol, centres[k], mats[i], box, grid)
            if w.mean() < min_inside:
                skipped += 1
                continue
            if normalize:
                m = w > 0
                s = sub[m].std()
                sub = np.where(m, (sub - sub[m].mean()) / (s if s > 0 else 1), 0)
            h = int(subset[i]) if subset is not None and subset[i] in (1, 2) else (1 if used % 2 == 0 else 2)
            sums[h] += sub
            wts[h] += w
            used += 1
            if progress:
                progress(used + skipped, len(ps))
    tot_w = wts[1] + wts[2]
    avg = ((sums[1] + sums[2]) / np.maximum(tot_w, 1e-6)).astype(np.float32)
    halves = tuple((sums[h] / np.maximum(wts[h], 1e-6)).astype(np.float32) for h in (1, 2))
    return AverageResult(avg, halves, used, skipped, apix_out)


def _iter_particles(ps, tomograms, box, rotate, normalize, min_inside):
    """Yield (row index, tomo name, subvolume, apix) for usable particles; None for skipped."""
    grid = box_grid(box)
    mats = ps.matrices if rotate else np.broadcast_to(np.eye(3), (len(ps), 3, 3))
    for tomo_name, idx in ps.df.groupby("rlnTomoName", sort=False).groups.items():
        if tomo_name not in tomograms:
            raise KeyError(f"no tomogram given for rlnTomoName={tomo_name!r}; have {list(tomograms)}")
        vol, apix = read_mrc(tomograms[tomo_name])
        centres = ps.centers_px[np.asarray(idx)] * (ps.ts_apix / apix)
        for k, i in enumerate(idx):
            sub, w = extract(vol, centres[k], mats[i], box, grid)
            if w.mean() < min_inside:
                yield None
                continue
            if normalize:
                m = w > 0
                s = sub[m].std()
                sub = np.where(m, (sub - sub[m].mean()) / (s if s > 0 else 1), 0).astype(np.float32)
            yield i, tomo_name, sub, apix, mats[i]


def _average_wedge(ps, tomograms, box, rotate, normalize, min_inside, progress, tilts, dose_weight, eps):
    from .wedge import rfft_grid, wedge_weights

    fshape = (box, box, box // 2 + 1)
    sF = {h: np.zeros(fshape, np.complex128) for h in (1, 2)}
    sW = {h: np.zeros(fshape, np.float64) for h in (1, 2)}
    nh = {1: 0, 2: 0}
    subset = ps.df["rlnRandomSubset"].to_numpy(int) if "rlnRandomSubset" in ps.df else None
    kgrid = rfft_grid(box)
    used = skipped = 0
    apix_out = None
    for item in _iter_particles(ps, tomograms, box, rotate, normalize, min_inside):
        if item is None:
            skipped += 1
            continue
        i, tomo, sub, apix, M = item
        apix_out = apix_out or apix
        if tomo not in tilts:
            raise KeyError(f"no tilt info for {tomo!r}")
        w = wedge_weights(box, tilts[tomo], M, apix, dose_weight, grid=kgrid)
        h = int(subset[i]) if subset is not None and subset[i] in (1, 2) else (1 if used % 2 == 0 else 2)
        sF[h] += w * np.fft.rfftn(sub)
        sW[h] += w * w
        nh[h] += 1
        used += 1
        if progress:
            progress(used + skipped, len(ps))

    def finish(F, W, n):
        return np.fft.irfftn(F / np.maximum(W, eps * max(n, 1)), s=(box,) * 3).astype(np.float32)

    avg = finish(sF[1] + sF[2], sW[1] + sW[2], used)
    halves = tuple(finish(sF[h], sW[h], nh[h]) for h in (1, 2))
    return AverageResult(avg, halves, used, skipped, apix_out)


def fsc(a, b, apix: float):
    """Fourier shell correlation. Returns (resolution_A[1:], fsc[1:])."""
    fa, fb = np.fft.rfftn(a), np.fft.rfftn(b)
    n = a.shape[0]
    kz = np.fft.fftfreq(n)[:, None, None]
    ky = np.fft.fftfreq(n)[None, :, None]
    kx = np.fft.rfftfreq(n)[None, None, :]
    shell = np.round(np.sqrt(kx**2 + ky**2 + kz**2) * n).astype(int)
    nsh = n // 2 + 1
    m = shell < nsh
    num = np.bincount(shell[m], (fa * np.conj(fb)).real[m], nsh)
    da = np.bincount(shell[m], (np.abs(fa) ** 2)[m], nsh)
    db = np.bincount(shell[m], (np.abs(fb) ** 2)[m], nsh)
    f = num / np.sqrt(np.maximum(da * db, 1e-30))
    k = np.arange(nsh)
    return (n * apix / np.maximum(k, 1))[1:], f[1:]


def resolution_at(res, f, threshold=0.143, mode: str = "first"):
    """Resolution where the FSC crosses ``threshold``.

    mode="first": first crossing (conservative).
    mode="last":  last crossing. Use this for data without CTF correction, where
    shared defocus produces FSC dips at CTF zeros that are not the resolution limit.
    """
    if mode == "last":
        above = np.nonzero(f >= threshold)[0]
        if len(above) == 0:
            return float(res[0])
        j = above[-1] + 1
        if j >= len(f):
            return float(res[-1])
    else:
        below = np.nonzero(f < threshold)[0]
        if len(below) == 0:
            return float(res[-1])
        j = below[0]
    if j == 0:
        return float(res[0])
    # linear interpolation in spatial frequency
    s0, s1 = 1 / res[j - 1], 1 / res[j]
    t = (f[j - 1] - threshold) / (f[j - 1] - f[j])
    return float(1 / (s0 + t * (s1 - s0)))


def tomograms_from_star(ps: ParticleSet, star_path, tomo_dir=None) -> dict:
    """{rlnTomoName: tomogram path} taken from the STAR file's rlnMicrographName.

    Each path is resolved in this order: ``tomo_dir/<name>`` if ``tomo_dir`` is
    given; otherwise as written (absolute, or relative to the current
    directory); otherwise relative to the STAR file's directory (RELION
    convention: paths relative to the project directory).
    """
    if "rlnMicrographName" not in ps.df:
        raise ValueError("STAR file has no rlnMicrographName column; pass tomogram files explicitly")
    star_dir = Path(star_path).resolve().parent
    pairs = ps.df[["rlnTomoName", "rlnMicrographName"]].astype(str).drop_duplicates()
    clash = pairs[pairs.duplicated("rlnTomoName", keep=False)]
    if not clash.empty:
        raise ValueError("several rlnMicrographName values share one rlnTomoName: "
                         + ", ".join(sorted(clash.rlnMicrographName)[:4]))
    out, missing = {}, []
    for tomo, name in pairs.itertuples(index=False):
        cands = [Path(tomo_dir) / name, Path(tomo_dir) / Path(name).name] if tomo_dir else \
                [Path(name), star_dir / name]
        hit = next((c for c in cands if c.is_file()), None)
        if hit is None:
            missing.append(name)
        else:
            out[tomo] = hit
    if missing:
        hint = (" (these are Warp .tomostar names, not tomogram files: put the reconstructed .mrc paths in "
                "rlnMicrographName or pass the tomograms on the command line)"
                if any(m.endswith(".tomostar") for m in missing) else "")
        looked = f"in {tomo_dir}" if tomo_dir else f"relative to {Path.cwd()} and {star_dir}"
        raise FileNotFoundError(f"{len(missing)} tomogram(s) from rlnMicrographName not found {looked}: "
                                + ", ".join(missing[:5]) + hint)
    return out


def map_tomograms(ps: ParticleSet, paths) -> dict:
    """Match tomogram files to rlnTomoName by stripped file name; one file + one tomo => direct."""
    names = list(dict.fromkeys(ps.df["rlnTomoName"]))
    paths = [Path(p) for p in paths]
    if len(paths) == 1 and len(names) == 1:
        return {names[0]: paths[0]}
    lookup = {tomo_name_from(p.name): p for p in paths}
    return {n: lookup[n] for n in names if n in lookup}
