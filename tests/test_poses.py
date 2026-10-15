import numpy as np
import pandas as pd
import pytest

from py3sta.align import align_maps
from py3sta.average import extract
from py3sta.geometry import euler_to_matrix
from py3sta.io.particles import ParticleSet
from py3sta.poses import apply_delta, chain, compose, invert
from py3sta.reference import make_mask, masked_cc

from test_align import APIX, BOX, chiral_object

TS_APIX = APIX / 2                     # coordinates in tilt-series pixels: tests the unit conversion
TOMO = 96


def _particles(n, seed=0):
    rng = np.random.default_rng(seed)
    e = np.column_stack([rng.uniform(-180, 180, n), rng.uniform(0, 180, n), rng.uniform(-180, 180, n)])
    e[0] = (30.0, 0.0, 40.0)                                                    # gimbal lock
    df = pd.DataFrame({"rlnTomoName": "TS_01",
                       "rlnCoordinateX": rng.uniform(80, 110, n), "rlnCoordinateY": rng.uniform(80, 110, n),
                       "rlnCoordinateZ": rng.uniform(80, 110, n),
                       "rlnOriginXAngst": rng.normal(0, 5, n), "rlnOriginYAngst": rng.normal(0, 5, n),
                       "rlnOriginZAngst": rng.normal(0, 5, n),
                       "rlnAngleRot": e[:, 0], "rlnAngleTilt": e[:, 1], "rlnAnglePsi": e[:, 2]})
    return ParticleSet(df, TS_APIX)


def test_chain_and_invert():
    rng = np.random.default_rng(1)
    M = euler_to_matrix(*rng.uniform(-180, 180, (3, 5)))
    c = rng.normal(size=(5, 3))
    R1, R2 = euler_to_matrix(10, 20, 30), euler_to_matrix(-50, 70, 5)
    t1, t2 = np.array([1.0, -2, 0.5]), np.array([0.3, 0.2, -1])
    a = compose(*compose(M, c, R1, t1), R2, t2)
    b = compose(M, c, *chain(R1, t1, R2, t2))
    np.testing.assert_allclose(a[0], b[0], atol=1e-12)
    np.testing.assert_allclose(a[1], b[1], atol=1e-12)
    back = compose(*compose(M, c, R1, t1), *invert(R1, t1))
    np.testing.assert_allclose(back[0], M, atol=1e-12)
    np.testing.assert_allclose(back[1], c, atol=1e-12)
    with pytest.raises(ValueError, match="mirrored"):
        compose(M, c, -np.eye(3), t1)


def test_apply_delta_roundtrip():
    """Known delta, then its inverse: original matrices and centres (incl. gimbal lock, per-particle deltas)."""
    ps = _particles(20)
    R, t = euler_to_matrix(25, 40, -60), np.array([2.0, -1.0, 0.5])
    moved = apply_delta(ps, R, t, APIX)
    np.testing.assert_allclose(ps.df["rlnCoordinateX"], moved.df["rlnCoordinateX"])         # coordinates kept
    back = apply_delta(moved, *invert(R, t), APIX)
    np.testing.assert_allclose(back.matrices, ps.matrices, atol=1e-10)
    np.testing.assert_allclose(back.centers_px, ps.centers_px, atol=1e-6)
    # centre moved by M'.T t, converted from box voxels to tilt-series pixels
    np.testing.assert_allclose(moved.centers_px - ps.centers_px,
                               np.einsum("nji,j->ni", moved.matrices, t) * APIX / TS_APIX, atol=1e-9)
    Rs = euler_to_matrix(*np.random.default_rng(2).uniform(-90, 90, (3, 5)))
    ts = np.arange(15.0).reshape(5, 3) / 10
    rows = np.zeros(20, bool)
    rows[[1, 4, 7, 9, 12]] = True
    part = apply_delta(ps, Rs, ts, APIX, rows=rows)
    np.testing.assert_allclose(part.matrices[~rows], ps.matrices[~rows], atol=1e-12)
    np.testing.assert_allclose(apply_delta(part, *invert(Rs, ts), APIX, rows=rows).matrices, ps.matrices, atol=1e-10)
    with pytest.raises(ValueError, match="per-particle"):
        apply_delta(ps, Rs, ts, APIX)


def _tomogram(ps):
    """Analytic tomogram (voxel APIX) with the chiral object at every particle's true pose."""
    tomo = np.zeros((TOMO,) * 3, np.float32)
    for M, c in zip(ps.matrices, ps.centers_px * TS_APIX / APIX):
        tomo += chiral_object(TOMO, APIX, shift_A=(c - TOMO // 2) * APIX, M=M)       # tomo(c + M.T r) = obj(r)
    return tomo


def _extract(tomo, ps, i):
    return extract(tomo, ps.centers_px[i] * TS_APIX / APIX, ps.matrices[i], BOX)[0]


def test_composed_pose_extracts_moved_map():
    """Extracting with the composed pose == the old subvolume moved by (R, t): obj(R.T (r + t))."""
    ps = _particles(1, seed=3)
    ps.df[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]] = 2 * TOMO / 2 + 0.6     # centre of the tomogram
    tomo = _tomogram(ps)
    m = make_mask(BOX, APIX, "sphere", [60.0], soft=10.0)
    assert masked_cc(_extract(tomo, ps, 0), chiral_object(), m) > 0.995
    R, t = euler_to_matrix(25, 40, -60), np.array([2.0, -1.5, 1.0])
    expected = chiral_object(BOX, APIX, shift_A=-t * APIX, M=R.T)                  # obj(R.T (r + t))
    assert masked_cc(_extract(tomo, apply_delta(ps, R, t, APIX), 0), expected, m) > 0.995
    assert masked_cc(_extract(tomo, ps, 0), expected, m) < 0.8                  # control: the move matters


def test_class_transform_recovered():
    """Two classes whose members have poses off by a class-specific (R, t): aligning the class average
    to the true structure and composing the result onto the members restores the true poses."""
    rng = np.random.default_rng(4)
    ps = _particles(4, seed=5)
    corners = np.array([[24, 24, 24], [72, 72, 24], [72, 24, 72], [24, 72, 72]], float)    # voxels, 48 apart
    ps.df[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]] = (corners + rng.uniform(-1, 1, (4, 3))) * APIX / TS_APIX
    ps.df[["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]] = 0.0
    tomo = _tomogram(ps)
    cls = np.array([1, 1, 2, 2])
    errors = {1: (euler_to_matrix(20, 30, 0), np.array([1.5, 0, -1])),
              2: (euler_to_matrix(-70, 50, 110), np.array([-1, 2, 0.5]))}
    given = ps
    for k, (R, t) in errors.items():
        given = apply_delta(given, *invert(R, t), APIX, rows=cls == k)
    m = make_mask(BOX, APIX, "sphere", [70.0], soft=12.0)
    fixed = chiral_object()
    fixed_pose = given
    for k in errors:
        avg = np.mean([_extract(tomo, given, i) for i in np.nonzero(cls == k)[0]], 0)
        res = align_maps(fixed, avg, APIX, m, lowpass=10.0, bin_factor=1, step=30.0, max_shift_A=24.0,
                         try_mirror=False)
        fixed_pose = apply_delta(fixed_pose, res.matrix, res.shift, APIX, rows=cls == k)
    ang = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", fixed_pose.matrices, ps.matrices) - 1) / 2, -1, 1)))
    assert ang.max() < 1.0, ang
    np.testing.assert_allclose(fixed_pose.centers_px * TS_APIX / APIX, ps.centers_px * TS_APIX / APIX, atol=0.2)
