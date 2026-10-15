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

from .io.mrc import read_header, read_mrc
from .io.particles import ParticleSet, tomo_name_from


def box_grid(box: int) -> np.ndarray:
    """(box, box, box, 3) xyz offsets from the centre voxel box//2, array order [z, y, x]."""
    r = np.arange(box, dtype=np.float32) - box // 2
    z, y, x = np.meshgrid(r, r, r, indexing="ij")
    return np.stack([x, y, z], -1)


def _trilinear(vol, pts):
    """Sample vol[z,y,x] at pts (N,3) xyz. Returns (values, inside-mask).

    Gathers from the flattened array with ``take`` (about 4x faster than 3-D fancy indexing).
    """
    nz, ny, nx = vol.shape
    flat = np.ascontiguousarray(vol, dtype=np.float32).ravel()
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    x0, y0, z0 = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = (x - x0).astype(np.float32), (y - y0).astype(np.float32), (z - z0).astype(np.float32)
    x0, y0, z0 = x0.astype(np.int64), y0.astype(np.int64), z0.astype(np.int64)
    inside = (x0 >= 0) & (y0 >= 0) & (z0 >= 0) & (x0 < nx - 1) & (y0 < ny - 1) & (z0 < nz - 1)
    np.clip(x0, 0, nx - 2, out=x0)
    np.clip(y0, 0, ny - 2, out=y0)
    np.clip(z0, 0, nz - 2, out=z0)
    i0 = (z0 * ny + y0) * nx + x0
    sy, sz = nx, nx * ny
    c00 = flat.take(i0);           c00 = c00 + fx * (flat.take(i0 + 1) - c00)
    c01 = flat.take(i0 + sy);      c01 = c01 + fx * (flat.take(i0 + sy + 1) - c01)
    c10 = flat.take(i0 + sz);      c10 = c10 + fx * (flat.take(i0 + sz + 1) - c10)
    c11 = flat.take(i0 + sz + sy); c11 = c11 + fx * (flat.take(i0 + sz + sy + 1) - c11)
    c0 = c00 + fy * (c01 - c00)
    c1 = c10 + fy * (c11 - c10)
    out = c0 + fz * (c1 - c0)
    out[~inside] = 0
    return out, inside


def _sample_points(grid, matrix, centre_xyz):
    g = grid.reshape(-1, 3)
    m = np.asarray(matrix, np.float32)
    # pts = g @ M + c, written element-wise: a BLAS matmul here spawns threads in every
    # worker process and oversubscribes the CPU (made -j 4 slower than -j 1).
    return g[:, :1] * m[0] + g[:, 1:2] * m[1] + g[:, 2:3] * m[2] + np.asarray(centre_xyz, np.float32)


def extract(tomo, centre_xyz, matrix, box: int, grid=None):
    """Return (subvolume[z,y,x], weight[z,y,x]) in the reference frame."""
    grid = box_grid(box) if grid is None else grid
    pts = _sample_points(grid, matrix, centre_xyz)
    shape = np.array(tomo.shape[::-1])
    lo = np.clip(np.floor(pts.min(0)).astype(int), 0, shape - 1)
    hi = np.clip(np.ceil(pts.max(0)).astype(int) + 2, 1, shape)
    if np.any(hi - lo < 2):
        z = np.zeros((box,) * 3, np.float32)
        return z, z
    crop = np.asarray(tomo[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]], dtype=np.float32)
    vals, inside = _trilinear(crop, pts - lo)
    return vals.reshape((box,) * 3), inside.reshape((box,) * 3).astype(np.float32)


def extract_crop(crop, lo, tomo_shape, centre_xyz, matrix, box: int, grid=None):
    """Like ``extract``, but sampling a stored crop whose first voxel is tomogram voxel ``lo`` (xyz).

    ``tomo_shape`` ([z, y, x]) decides what lies inside the tomogram, so the result equals
    ``extract(tomo, ...)`` wherever the sample points fall inside the crop (always within the
    extraction sphere while the particle stays within the crop margin, see stack.py).
    """
    grid = box_grid(box) if grid is None else grid
    pts = _sample_points(grid, matrix, centre_xyz)
    p0 = np.floor(pts)
    in_tomo = ((p0 >= 0) & (p0 < np.array(tomo_shape[::-1]) - 1)).all(1)
    vals, in_crop = _trilinear(np.asarray(crop, np.float32), pts - np.asarray(lo, np.float32))
    w = in_tomo & in_crop
    vals[~w] = 0
    return vals.reshape((box,) * 3), w.reshape((box,) * 3).astype(np.float32)


def extraction_sphere(box: int) -> np.ndarray:
    """Subvolumes are defined inside this sphere only (radius box//2 - 1 voxels): a stack crop of
    box + 2*margin covers it for any rotation, so averages from the stack and from the tomogram agree."""
    return sphere_mask(box, box // 2 - 1)


def sphere_mask(box: int, radius_vox: float) -> np.ndarray:
    r = np.sqrt((box_grid(box) ** 2).sum(-1))
    return (r <= radius_vox).astype(np.float32)


def prepare_subvolume(sub, w, sphere, norm_mask, normalize: bool):
    """Restrict to the extraction sphere and normalise to mean 0 / std 1 over the voxels inside both
    the tomogram and the particle-radius sphere (``norm_mask``). Returns (sub, w, inside_fraction)."""
    w = w * sphere
    inside = float(w.sum() / sphere.sum())
    if normalize:
        m = (w > 0) & (norm_mask > 0)
        v = sub[m]
        s = v.std() if v.size else 0.0
        mean = v.mean() if v.size else 0.0
        sub = np.where(w > 0, (sub - mean) / (s if s > 0 else 1), 0).astype(np.float32)
    else:
        sub = (sub * w).astype(np.float32)
    return sub, w, inside


@dataclass
class AverageResult:
    average: np.ndarray
    halves: tuple | None
    n_used: int
    n_skipped: int
    apix: float
    # {rlnTomoName: (sum, weight sum, n_used)} over both halves; set with per_tomo=True
    per_tomo: dict | None = None
    fourier: bool = False
    wiener_eps: float = 0.05


def half_sets(ps: ParticleSet) -> np.ndarray:
    """Half-set (1/2) per particle: rlnRandomSubset if present, else alternating by row."""
    alt = np.arange(len(ps)) % 2 + 1
    if "rlnRandomSubset" in ps.df:
        s = ps.df["rlnRandomSubset"].to_numpy()
        return np.where(np.isin(s, (1, 2)), s, alt).astype(int)
    return alt


def _open_source(src, box, grid):
    """Worker side: (sampler(k, centre, M) -> (sub, w), tomogram shape [z,y,x], apix)."""
    if src[0] == "tomo":
        vol, apix = read_mrc(src[1])
        return (lambda k, c, M: extract(vol, c, M, box, grid)), vol.shape, apix
    _, path, size, rows, lo, shape, apix = src
    data = _stack_memmap(path, size)
    return (lambda k, c, M: extract_crop(data[rows[k]], lo[k], shape, c, M, box, grid)), tuple(shape), apix


def _stack_memmap(path, size):
    from .io.mrc import open_volume_stack

    return open_volume_stack(path, 0, size, mode="r")


def _accumulate(src, centres, mats, halves, box, normalize, min_inside, tilt, dose_weight,
                ctf_mode=None, height_sign=1.0, norm_radius=None):
    """Worker: partial sums for one chunk of particles of one tomogram (runs in a subprocess).

    src: ("tomo", path) or ("stack", path, crop size, rows, crop lo (n,3), tomogram shape, apix).
    """
    grid = box_grid(box)
    sample, tshape, apix = _open_source(src, box, grid)
    sphere = extraction_sphere(box)
    nmask = sphere if norm_radius is None else sphere_mask(box, norm_radius)
    if tilt is not None:
        from .wedge import particle_defocus, rfft_grid, wedge_weights

        kgrid = rfft_grid(box)
        tomo_centre = (np.array(tshape[::-1], float) - 1) / 2
        shp, ctype = (box, box, box // 2 + 1), np.complex128
    else:
        shp, ctype = (box,) * 3, np.float64
    S = {h: np.zeros(shp, ctype) for h in (1, 2)}
    W = {h: np.zeros(shp, np.float64) for h in (1, 2)}
    n = {1: 0, 2: 0}
    skipped = 0
    for k, (c, M, h) in enumerate(zip(centres, mats, halves)):
        sub, w = sample(k, c, M)
        sub, w, inside = prepare_subvolume(sub, w, sphere, nmask, normalize)
        if inside < min_inside:
            skipped += 1
            continue
        if tilt is not None:
            dfc = particle_defocus(tilt, (np.asarray(c) - tomo_centre) * apix, height_sign) if ctf_mode else None
            wt = wedge_weights(box, tilt, M, apix, dose_weight, grid=kgrid, defocus_A=dfc, ctf_mode=ctf_mode)
            S[h] += wt * np.fft.rfftn(sub)
            W[h] += wt * wt
        else:
            S[h] += sub
            W[h] += w
        n[h] += 1
    return S, W, n, skipped, len(centres), apix


def average_particles(ps: ParticleSet, tomograms: dict | None, box: int, rotate: bool = True,
                      normalize: bool = True, min_inside: float = 0.9, progress=None,
                      tilts: dict | None = None, dose_weight: bool = False,
                      wiener_eps: float = 0.05, jobs: int = 1, chunk: int | None = None,
                      ctf_mode: str | None = None, height_sign: float = 1.0,
                      norm_radius: float | None = None, stack=None, per_tomo: bool = False) -> AverageResult:
    """Global average of all particles.

    tomograms: {rlnTomoName: path}; or ``stack`` (stack.ParticleStack): particles are sampled
    from their stored crops instead (same result, see extract_crop). Subvolumes are defined inside
    the extraction sphere (radius box//2 - 1); particles with < ``min_inside`` of that sphere inside
    the tomogram are skipped. normalize: mean 0 / std 1 over the voxels within ``norm_radius``
    (voxels; default: the extraction sphere). Half maps follow rlnRandomSubset (else alternate rows).

    tilts: {rlnTomoName: TiltInfo}. If given, the average is wedge-weighted in
    Fourier space:  F = sum(w_i F_i) / max(sum(w_i^2), wiener_eps * n),
    with w_i the particle's tilt coverage rotated into the reference frame.

    ctf_mode: None, "flip" (phase flipping) or "wiener" (signed CTF in the
    weights, so the denominator becomes sum(w^2 CTF^2)); needs tilts with CTF
    info. height_sign: direction of the per-particle defocus offset (ctf-check).

    jobs: number of worker processes (each reads its own chunk of particles).
    Results agree across ``jobs`` to float rounding. per_tomo: also return per-tomogram sums.
    """
    from concurrent.futures import ProcessPoolExecutor

    if ctf_mode and tilts is None:
        raise ValueError("CTF correction needs tilt information with CTF (Warp XML)")
    halves_all = half_sets(ps)
    mats_all = ps.matrices if rotate else np.broadcast_to(np.eye(3), (len(ps), 3, 3))
    jobs = max(1, int(jobs))
    loc = stack.locate(ps) if stack is not None else None
    tasks, task_tomo = [], []
    for tomo, idx in ps.df.groupby("rlnTomoName", sort=False).groups.items():
        if loc is None and tomo not in tomograms:
            raise KeyError(f"no tomogram given for rlnTomoName={tomo!r}; have {list(tomograms)}")
        if tilts is not None and tomo not in tilts:
            raise KeyError(f"no tilt info for {tomo!r}")
        idx = np.asarray(idx)
        apix = stack.apix if loc is not None else read_header(tomograms[tomo])["apix"]
        centres = ps.centers_px[idx] * (ps.ts_apix / apix)
        size = chunk or max(1, -(-len(idx) // (jobs * 4)))
        for s in range(0, len(idx), size):
            sl = slice(s, s + size)
            if loc is None:
                src = ("tomo", str(tomograms[tomo]))
            else:
                src = ("stack", str(stack.data_path), stack.size, loc.rows[idx[sl]], loc.lo[idx[sl]],
                       loc.tomo_shape[tomo], apix)
            tasks.append((src, centres[sl], np.ascontiguousarray(mats_all[idx[sl]]),
                          halves_all[idx[sl]], box, normalize, min_inside,
                          None if tilts is None else tilts[tomo], dose_weight, ctf_mode, height_sign, norm_radius))
            task_tomo.append(tomo)

    S = W = None
    nh = {1: 0, 2: 0}
    skipped = done = 0
    apix_out = None
    by_tomo = {} if per_tomo else None

    def add(res, tomo):
        nonlocal S, W, skipped, done, apix_out
        s, w, n, sk, cnt, apix = res
        if S is None:
            S, W = {h: s[h].copy() for h in (1, 2)}, {h: w[h].copy() for h in (1, 2)}
        else:
            for h in (1, 2):
                S[h] += s[h]
                W[h] += w[h]
        for h in (1, 2):
            nh[h] += n[h]
        if by_tomo is not None:
            st, wt, nt = by_tomo.get(tomo, (0, 0, 0))
            by_tomo[tomo] = (st + s[1] + s[2], wt + w[1] + w[2], nt + n[1] + n[2])
        skipped += sk
        done += cnt
        apix_out = apix_out or apix
        if progress:
            progress(done, len(ps))

    if jobs == 1 or len(tasks) == 1:
        for t, tomo in zip(tasks, task_tomo):
            add(_accumulate(*t), tomo)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for res, tomo in zip(ex.map(_accumulate, *zip(*tasks)), task_tomo):  # ordered -> deterministic sums
                add(res, tomo)

    used = nh[1] + nh[2]
    fin = _finisher(box, tilts is not None, wiener_eps)
    avg = fin(S[1] + S[2], W[1] + W[2], used)
    halves = (fin(S[1], W[1], nh[1]), fin(S[2], W[2], nh[2]))
    return AverageResult(avg, halves, used, skipped, apix_out, by_tomo, tilts is not None, wiener_eps)


def _finisher(box, fourier: bool, wiener_eps: float):
    if fourier:
        def fin(F, Wt, n):
            return np.fft.irfftn(F / np.maximum(Wt, wiener_eps * max(n, 1)), s=(box,) * 3,
                                 axes=(0, 1, 2)).astype(np.float32)
    else:
        def fin(Sr, Wt, n):
            return (Sr / np.maximum(Wt, 1e-6)).astype(np.float32)
    return fin


def per_tomo_cc(res: AverageResult, radius_vox: float, lowpass_A: float | None = None, min_particles: int = 5):
    """Each tomogram's average vs the average of all other tomograms (leave-one-out), Pearson CC inside
    a sphere of ``radius_vox`` after low-pass. Returns a DataFrame (tomo, n, cc, snr, z, flagged).

    CC grows with the number of particles, so tomograms are compared by the per-particle SNR it implies,
    snr = cc² / (n (1 - cc²)); flagged = robust z-score of log(snr) < -3 (median / 1.4826·MAD over
    tomograms). Tomograms with < min_particles get NaN and are never flagged."""
    import pandas as pd

    if not res.per_tomo:
        raise ValueError("average was computed without per_tomo=True")
    box = res.average.shape[0]
    fin = _finisher(box, res.fourier, res.wiener_eps)
    S_all = sum(v[0] for v in res.per_tomo.values())
    W_all = sum(v[1] for v in res.per_tomo.values())
    n_all = sum(v[2] for v in res.per_tomo.values())
    m = sphere_mask(box, radius_vox) > 0
    filt = None
    if lowpass_A:
        f = np.sqrt(sum(g ** 2 for g in np.meshgrid(np.fft.fftfreq(box), np.fft.fftfreq(box),
                                                     np.fft.rfftfreq(box), indexing="ij"))) / res.apix
        filt = (f <= 1.0 / lowpass_A).astype(np.float32)

    def lp(v):
        return v if filt is None else np.fft.irfftn(np.fft.rfftn(v) * filt, s=v.shape, axes=(0, 1, 2))

    rows = []
    for tomo, (s, w, n) in res.per_tomo.items():
        cc = np.nan
        if n >= min_particles and n_all - n >= min_particles:
            a, b = lp(fin(s, w, n))[m], lp(fin(S_all - s, W_all - w, n_all - n))[m]
            a, b = a - a.mean(), b - b.mean()
            den = np.sqrt((a * a).sum() * (b * b).sum())
            cc = float((a * b).sum() / den) if den > 0 else np.nan
        rows.append({"tomo": tomo, "n": int(n), "cc": cc})
    df = pd.DataFrame(rows)
    cc2 = df["cc"].clip(lower=0, upper=1 - 1e-9) ** 2
    df["snr"] = cc2 / (df["n"] * (1 - cc2))
    ok = df["snr"].notna()
    with np.errstate(divide="ignore"):
        logs = np.log(df["snr"])
    df["z"] = np.nan
    if ok.sum() >= 3:
        med = logs[ok & (df["snr"] > 0)].median()
        mad = (logs[ok & (df["snr"] > 0)] - med).abs().median()
        df.loc[ok, "z"] = (logs[ok] - med) / (1.4826 * mad) if mad > 0 else 0.0
    df["flagged"] = ok & (df["z"] < -3)
    return df


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


def tomograms_from_star(ps: ParticleSet, star_path) -> dict:
    """{rlnTomoName: tomogram path} from rlnMicrographName.

    Paths are used as written (absolute, or relative to the current directory),
    else relative to the STAR file's directory (RELION project convention).
    """
    try:
        return per_tomo_files(ps, star_path, "rlnMicrographName")
    except FileNotFoundError as e:
        if ".tomostar" in str(e):
            raise FileNotFoundError(str(e) + " (these are Warp .tomostar names, not tomogram files: put the "
                                    "reconstructed .mrc paths in rlnMicrographName or pass the tomograms "
                                    "on the command line)") from None
        raise


def per_tomo_files(ps: ParticleSet, star_path, column: str) -> dict:
    """{rlnTomoName: Path} for a per-tomogram file column (rlnMicrographName, rlnWarpXMLfile, ...)."""
    if column not in ps.df:
        raise ValueError(f"STAR file has no {column} column")
    star_dir = Path(star_path).resolve().parent
    pairs = ps.df[["rlnTomoName", column]].astype(str).drop_duplicates()
    clash = pairs[pairs.duplicated("rlnTomoName", keep=False)]
    if not clash.empty:
        raise ValueError(f"several {column} values share one rlnTomoName: " + ", ".join(sorted(clash[column])[:4]))
    out, missing = {}, []
    for tomo, name in pairs.itertuples(index=False):
        hit = next((c for c in (Path(name), star_dir / name) if c.is_file()), None)
        if hit is None:
            missing.append(name)
        else:
            out[tomo] = hit
    if missing:
        raise FileNotFoundError(f"{len(missing)} file(s) from {column} not found relative to {Path.cwd()} "
                                f"or {star_dir}: " + ", ".join(missing[:5]))
    return out


def map_tomograms(ps: ParticleSet, paths) -> dict:
    """Match tomogram files to rlnTomoName by stripped file name; one file + one tomo => direct."""
    names = list(dict.fromkeys(ps.df["rlnTomoName"]))
    paths = [Path(p) for p in paths]
    if len(paths) == 1 and len(names) == 1:
        return {names[0]: paths[0]}
    lookup = {tomo_name_from(p.name): p for p in paths}
    return {n: lookup[n] for n in names if n in lookup}
