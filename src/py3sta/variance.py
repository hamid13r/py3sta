"""Voxel-wise variance of the aligned particles: where is the stack heterogeneous?

Each particle is sampled at its current pose and normalised (mean 0, std 1) inside a *normalisation mask*
-- a part believed to be homogeneous (e.g. the membrane of an ATP synthase) -- so every particle has the
same contrast there. The variance over particles is then computed for every voxel (msa.ParticleVolumes):

* ``wmd`` (default) -- variance of the wedge-masked differences W_i (F_i - A): the missing wedge of each
  particle contributes nothing, so the map shows structural variability, not wedge orientation.
* ``plain`` -- variance of x_i (or W_i x_i with ``wedge``); dominated by the wedge pattern on real data.

Statistics are reported inside an optional *variance mask* (default: the extraction sphere), inside the
normalisation mask, and outside both (the solvent / noise baseline).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .average import extraction_sphere
from .msa import MSASettings, ParticleVolumes, binned_mask


@dataclass
class VarianceResult:
    variance: np.ndarray                    # (b, b, b) over particles, zero outside the extraction sphere
    mean: np.ndarray                        # (b, b, b) mean of the data volumes (wmd: ~0; plain: the average)
    average: np.ndarray                     # (b, b, b) wmd: Wiener average; plain: = mean
    n: int
    box: int                                # binned box
    apix: float                             # binned voxel size


def variance_map(ps, stack, norm_mask, apix: float, settings: MSASettings, tilts: dict | None = None,
                 invert: bool = False, device=None, chunk: int = 64, progress=None) -> VarianceResult:
    """Voxel-wise variance (1 / (n - 1)) of the particles of ``ps`` (see module doc). norm_mask: (box^3)."""
    box = np.asarray(norm_mask).shape[0]
    pv = ParticleVolumes(ps, stack, box, apix, settings, norm_mask, tilts, invert, device, chunk, progress)
    if pv.n < 2:
        raise ValueError(f"variance needs at least 2 particles (got {pv.n})")
    shape = (pv.b,) * 3
    s1 = np.zeros(shape, np.float64)
    s2 = np.zeros(shape, np.float64)
    for _, vols in pv:
        v = vols.astype(np.float64)
        s1 += v.sum(0)
        s2 += (v * v).sum(0)
    n = pv.n
    var = np.maximum(s2 - s1 * s1 / n, 0) / (n - 1)
    sphere = extraction_sphere(pv.b)
    return VarianceResult((var * sphere).astype(np.float32), (s1 / n).astype(np.float32), pv.average, n, pv.b,
                          pv.apix)


def region_stats(res: VarianceResult, norm_mask, apix: float, bin: int, mask=None) -> dict:
    """Variance statistics inside the variance mask (or the extraction sphere), the normalisation mask
    and outside both (masks at full sampling; a voxel is inside where the binned mask is > 0.5)."""
    sphere = extraction_sphere(res.box) > 0
    nm = (binned_mask(norm_mask, apix, bin) > 0.5) & sphere
    vm = (binned_mask(mask, apix, bin) > 0.5) & sphere if mask is not None else sphere
    regions = {"variance_mask": vm, "norm_mask": nm, "outside": sphere & ~nm & ~vm}

    def stats(m):
        v = res.variance[m]
        if not v.size:
            return {"voxels": 0}
        return {"voxels": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    out = {k: stats(m) for k, m in regions.items()}
    base = out["outside"].get("median") or out["norm_mask"].get("median")
    if base:
        for k in ("variance_mask", "norm_mask"):
            if out[k].get("voxels"):
                out[k]["median_over_baseline"] = out[k]["median"] / base
    out["baseline"] = "outside" if out["outside"].get("median") else "norm_mask"
    return out
