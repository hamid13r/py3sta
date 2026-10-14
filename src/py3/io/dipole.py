"""Dipole picks (.pos) -> ParticleSet. Replaces pos2i3.py.

A ``.pos`` file has 3 columns (x y z) with two consecutive rows per particle:
the first and second click of the dipole. The particle centre is the midpoint;
the normal points from pick 1 to pick 2 and is aligned with reference +z.

Differences from pos2i3.py:
  * centres keep sub-voxel precision (pos2i3 truncated with int(), a bias of
    up to one voxel toward the origin, magnified by --scale);
  * any whitespace is accepted; odd row counts raise a clear error;
  * the random in-plane angle is a clean uniform rotation about the normal;
  * tilt/psi priors are written (rlnAngleTiltPrior/rlnAnglePsiPrior), so
    RELION can restrain the membrane normal with --sigma_tilt/--sigma_psi.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..geometry import frame_from_normal, matrix_to_euler
from .particles import DEFAULT_TOMO_REGEX, ParticleSet, tomo_name_from


def read_pos(path) -> np.ndarray:
    rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    a = np.asarray(rows, dtype=float)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError(f"{path}: expected >=3 columns of x y z")
    if len(a) % 2:
        raise ValueError(f"{path}: odd number of picks ({len(a)}); dipoles need pairs")
    return a[:, :3]


def dipoles_to_particles(pos_files, pick_apix: float, ts_apix: float | None = None,
                         random_inplane: bool = False, seed: int | None = None,
                         strip_regex: str | None = DEFAULT_TOMO_REGEX, **kw) -> ParticleSet:
    ts_apix = ts_apix or pick_apix
    s = pick_apix / ts_apix
    rng = np.random.default_rng(seed)
    frames = []
    for p in pos_files:
        a = read_pos(p)
        p1, p2 = a[0::2], a[1::2]
        v = p2 - p1
        bad = np.linalg.norm(v, axis=1) == 0
        if bad.any():
            raise ValueError(f"{p}: {int(bad.sum())} dipoles have identical picks")
        inplane = rng.uniform(0, 360, len(v)) if random_inplane else np.zeros(len(v))
        m = frame_from_normal(v, inplane)
        eul = matrix_to_euler(m)
        centre = 0.5 * (p1 + p2) * s
        frames.append(pd.DataFrame({
            "rlnTomoName": tomo_name_from(Path(p).stem, strip_regex),
            "rlnCoordinateX": centre[:, 0], "rlnCoordinateY": centre[:, 1], "rlnCoordinateZ": centre[:, 2],
            "rlnAngleRot": eul[:, 0], "rlnAngleTilt": eul[:, 1], "rlnAnglePsi": eul[:, 2],
            "rlnAngleTiltPrior": eul[:, 1], "rlnAnglePsiPrior": eul[:, 2],
        }))
    if not frames:
        raise ValueError("no .pos files given")
    return ParticleSet(pd.concat(frames, ignore_index=True), ts_apix, **kw)
