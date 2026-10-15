import os

import numpy as np
import pandas as pd
import pytest

from py3sta.average import average_particles, extract, extract_crop, extraction_sphere
from py3sta.geometry import euler_to_matrix
from py3sta.io.mrc import write_mrc
from py3sta.io.particles import ParticleSet
from py3sta.stack import ParticleStack, StackError, crop_size, max_drift, refresh_stack

rng = np.random.default_rng(3)


def _tomos(tmp_path, n=2, shape=(40, 64, 72)):
    paths = {}
    for k in range(n):
        p = tmp_path / f"TS_{k:02d}.mrc"
        write_mrc(p, rng.normal(size=shape).astype(np.float32), 2.0)
        paths[f"TS_{k:02d}"] = p
    return paths


def _particles(n_per=6, tomos=("TS_00", "TS_01"), shape=(40, 64, 72), edge=False):
    rows = []
    for t in tomos:
        for i in range(n_per):
            c = rng.uniform([20, 20, 14], [shape[2] - 20, shape[1] - 20, shape[0] - 14])
            if edge and i == 0:
                c[2] = 3.0                           # near the tomogram edge: zero padded crop
            rot, tilt, psi = rng.uniform(-180, 180), rng.uniform(0, 180), rng.uniform(-180, 180)
            rows.append(dict(rlnTomoName=t, rlnTomoParticleName=f"{t}/{i + 1}", rlnCoordinateX=c[0] * 2,
                             rlnCoordinateY=c[1] * 2, rlnCoordinateZ=c[2] * 2, rlnAngleRot=rot,
                             rlnAngleTilt=tilt, rlnAnglePsi=psi, rlnRandomSubset=1 + i % 2))
    return ParticleSet(pd.DataFrame(rows), ts_apix=1.0)       # coordinates at 1 Å, tomograms at 2 Å


@pytest.mark.parametrize("box,margin", [(16, 4), (15, 3), (12, 1)])
def test_crop_covers_sphere_up_to_max_drift(box, margin):
    """Within max_drift, sampling the crop == sampling the tomogram inside the extraction sphere."""
    vol = rng.normal(size=(50, 50, 50)).astype(np.float32)
    size = crop_size(box, margin)
    sphere = extraction_sphere(box) > 0
    for _ in range(30):
        cc = np.array([25, 25, 25])
        lo = cc - size // 2
        crop = vol[lo[2]:lo[2] + size, lo[1]:lo[1] + size, lo[0]:lo[0] + size]
        d = rng.uniform(-max_drift(margin), max_drift(margin), 3)
        d[rng.integers(3)] = max_drift(margin) * rng.choice([-1, 1])      # at the limit on one axis
        M = euler_to_matrix(*rng.uniform(-180, 180, 3))
        a, wa = extract(vol, cc + d, M, box)
        b, wb = extract_crop(crop, lo, vol.shape, cc + d, M, box)
        assert wb[sphere].min() == 1
        np.testing.assert_array_equal(a[sphere], b[sphere])


def test_build_and_average_equal(tmp_path):
    tomos = _tomos(tmp_path)
    ps = _particles(edge=True)
    rep = refresh_stack(tmp_path / "stack", ps, tomos, box=16, margin=4, apix=2.0, dtype="float32")
    assert rep.rebuilt and rep.n_new == len(ps) and rep.n_rows == len(ps)
    st = ParticleStack.open(tmp_path / "stack")
    assert st.data().shape == (len(ps), 24, 24, 24)
    # a crop is a plain slice of the tomogram (zero padded at the edge)
    from py3sta.io.mrc import read_mrc

    vol, _ = read_mrc(tomos["TS_00"])
    loc = st.locate(ps)
    r, lo = loc.rows[1], loc.lo[1]
    np.testing.assert_array_equal(st.data()[r], vol[lo[2]:lo[2] + 24, lo[1]:lo[1] + 24, lo[0]:lo[0] + 24])
    assert loc.lo[0][2] < 0 and st.data()[loc.rows[0]][0].max() == 0        # edge particle: padded
    for kw in ({}, {"tilts": None, "norm_radius": 5.0}):
        a = average_particles(ps, tomos, 16, min_inside=0.0, **kw)
        b = average_particles(ps, None, 16, min_inside=0.0, stack=st, **kw)
        assert a.n_used == b.n_used == len(ps)
        np.testing.assert_allclose(b.average, a.average, atol=1e-5)


def test_float16_close_to_float32(tmp_path):
    tomos = _tomos(tmp_path)
    ps = _particles()
    refresh_stack(tmp_path / "s16", ps, tomos, box=16, margin=4, apix=2.0)
    a = average_particles(ps, tomos, 16).average
    b = average_particles(ps, None, 16, stack=ParticleStack.open(tmp_path / "s16")).average
    assert np.abs(b - a).max() < 2e-3 * np.abs(a).max()


def test_refresh_incremental(tmp_path):
    tomos = _tomos(tmp_path)
    ps = _particles()
    root = tmp_path / "stack"
    refresh_stack(root, ps, tomos, 16, 4, 2.0)
    data0 = np.array(ParticleStack.open(root).data())
    rep = refresh_stack(root, ps, tomos, 16, 4, 2.0)
    assert not rep.changed                                         # nothing to do
    # pose change only: still valid
    ps.df["rlnAngleRot"] += 30
    assert not refresh_stack(root, ps, tomos, 16, 4, 2.0).changed
    # drift: particle 0 moves 3 voxels (= max_drift) -> kept; particle 1 moves 4 -> re-cropped
    ps.df.loc[0, "rlnOriginXAngst"] = -3 * 2.0
    ps.df.loc[1, "rlnOriginXAngst"] = -4 * 2.0
    rep = refresh_stack(root, ps, tomos, 16, 4, 2.0)
    assert (rep.n_drift, rep.n_tomogram, rep.n_new) == (1, 0, 0)
    st = ParticleStack.open(root)
    st.locate(ps)                                                  # all within the margin now
    with pytest.raises(StackError, match="drifted"):
        st.locate(ps, shift_limit=2)
    data1 = np.array(st.data())
    changed = [i for i in range(len(ps)) if not np.array_equal(data0[i], data1[i])]
    assert changed == [1]
    # new particle appended; replaced tomogram re-crops only its rows
    extra = _particles(n_per=1, tomos=("TS_01",))
    extra.df["rlnTomoParticleName"] = "TS_01/99"
    ps2 = ParticleSet(pd.concat([ps.df, extra.df], ignore_index=True), 1.0)
    write_mrc(tomos["TS_01"], rng.normal(size=(40, 64, 72)).astype(np.float32), 2.0)
    os.utime(tomos["TS_01"], ns=(1, 1))
    rep = refresh_stack(root, ps2, tomos, 16, 4, 2.0)
    assert (rep.n_new, rep.n_tomogram, rep.changed_tomograms) == (1, 6, ["TS_01"])
    assert ParticleStack.open(root).data().shape[0] == len(ps2)
    # changing the margin rebuilds
    assert refresh_stack(root, ps2, tomos, 16, 5, 2.0).rebuilt


def test_refresh_errors(tmp_path):
    tomos = _tomos(tmp_path, n=1)
    ps = _particles(tomos=("TS_00",))
    with pytest.raises(StackError, match="margin must be >= 1"):
        refresh_stack(tmp_path / "s", ps, tomos, 16, 0, 2.0)
    with pytest.raises(StackError, match="voxel size"):
        refresh_stack(tmp_path / "s", ps, tomos, 16, 4, 3.0)
    ps.df.loc[1, "rlnTomoParticleName"] = ps.df.loc[0, "rlnTomoParticleName"]
    with pytest.raises(StackError, match="not unique"):
        refresh_stack(tmp_path / "s", ps, tomos, 16, 4, 2.0)


def test_torch_stack_parity(tmp_path):
    torch = pytest.importorskip("torch")
    from py3sta.backend_torch import average_particles_torch

    tomos = _tomos(tmp_path)
    ps = _particles(edge=True)
    refresh_stack(tmp_path / "stack", ps, tomos, 16, 4, 2.0, dtype="float32")
    st = ParticleStack.open(tmp_path / "stack")
    a = average_particles(ps, None, 16, stack=st, norm_radius=6.0, per_tomo=True)
    b = average_particles_torch(ps, None, 16, stack=st, norm_radius=6.0, per_tomo=True, device="cpu", batch=4)
    c = average_particles_torch(ps, tomos, 16, norm_radius=6.0, device="cpu")
    assert a.n_used == b.n_used == c.n_used
    np.testing.assert_allclose(b.average, a.average, atol=1e-5)
    np.testing.assert_allclose(c.average, a.average, atol=1e-5)
    for t in a.per_tomo:
        np.testing.assert_allclose(b.per_tomo[t][0], a.per_tomo[t][0], atol=1e-3)
        assert b.per_tomo[t][2] == a.per_tomo[t][2]
