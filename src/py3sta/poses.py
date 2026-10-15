"""Pose composition (PLAN.md §3): one function for every pose change (particle alignment, class transforms).

A pose change ("delta") is a rotation ``R`` and a shift ``t`` (voxels at the box pixel size), in the
reference frame, with the meaning of ``align.apply_alignment``: the new subvolume is the old one sampled at
``R.T (r + t)``. With extraction ``sub(r) = tomo(c + M.T r)`` this gives

    M' = R M,        c' = c + M'.T t.

Deltas chain as ``(R2, t2) after (R1, t1) = (R2 R1, R2 t1 + t2)`` and invert as ``(R.T, -R.T t)``.
``align_maps(fixed, class_average)`` returns ``(res.matrix, res.shift)`` in exactly this form.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .geometry import matrix_to_euler
from .io.particles import ANG, ORI, ParticleSet


def _check(R) -> np.ndarray:
    R = np.asarray(R, float)
    if np.any(np.linalg.det(R) < 0):
        raise ValueError("a mirrored transform (det < 0) cannot be applied to particle poses")
    return R


def compose(M, c, R, t) -> tuple[np.ndarray, np.ndarray]:
    """New (matrices, centres) after the delta (R, t). M (n, 3, 3); c (n, 3) voxels;
    R (3, 3) or (n, 3, 3); t (3,) or (n, 3) voxels -- c and t in the same voxel units."""
    M2 = _check(R) @ np.asarray(M, float)
    c2 = np.asarray(c, float) + np.einsum("...ji,...j->...i", M2, np.asarray(t, float))
    return M2, c2


def chain(R1, t1, R2, t2) -> tuple[np.ndarray, np.ndarray]:
    """The delta equal to applying (R1, t1) then (R2, t2)."""
    R2 = np.asarray(R2, float)
    return R2 @ np.asarray(R1, float), np.einsum("...ij,...j->...i", R2, np.asarray(t1, float)) + t2


def invert(R, t) -> tuple[np.ndarray, np.ndarray]:
    Rt = np.swapaxes(np.asarray(R, float), -1, -2)
    return Rt, -np.einsum("...ij,...j->...i", Rt, np.asarray(t, float))


def apply_delta(ps: ParticleSet, R, t, apix: float, rows=None) -> ParticleSet:
    """Copy of ``ps`` with the delta applied to ``rows`` (boolean mask or indices; default all).

    t: voxels at ``apix`` (the box pixel size). New angles from the composed matrix; the shift goes into
    rlnOriginX/Y/ZAngst (coordinates unchanged, as RELION), so true centre = coord - origin / ts_apix.
    """
    df = ps.df.copy()
    idx = np.arange(len(df)) if rows is None else np.arange(len(df))[np.asarray(rows)]
    R, t = np.asarray(R, float), np.asarray(t, float)
    if R.ndim == 3 and len(R) != len(idx) or t.ndim == 2 and len(t) != len(idx):
        raise ValueError(f"per-particle deltas: {len(R) if R.ndim == 3 else len(t)} given for {len(idx)} rows")
    M = ps.matrices[idx]
    origin = df[ORI].to_numpy(float)[idx]
    M2, o2 = compose(M, origin, R, -t * apix)               # origin' = origin - M'.T t apix  (Å)
    df.loc[df.index[idx], ANG] = matrix_to_euler(M2)
    df.loc[df.index[idx], ORI] = o2
    return dataclasses.replace(ps, df=df, meta=dict(ps.meta))
