import numpy as np
import pandas as pd

from py3sta.average import average_particles, extract
from py3sta.geometry import euler_to_matrix
from py3sta.io.mrc import read_mrc, write_mrc
from py3sta.io.particles import ParticleSet


def test_identity_extract_equals_crop():
    rng = np.random.default_rng(1)
    vol = rng.normal(size=(40, 50, 60)).astype(np.float32)
    sub, w = extract(vol, (30, 25, 20), np.eye(3), 16)
    np.testing.assert_allclose(sub, vol[12:28, 17:33, 22:38], atol=1e-5)
    assert w.min() == 1


def test_rotation_convention_and_average(tmp_path):
    """Place a rotated asymmetric object; the average must recover it in the reference frame."""
    box = 24
    ref = np.zeros((box,) * 3, np.float32)
    ref[12, 12, 12:20] = 1  # +x arm (long)
    ref[12, 12:16, 12] = 1  # +y arm (short)
    vol = np.zeros((48, 48, 96), np.float32)
    eul = [(30, 60, -45), (120, 100, 10)]
    rows = []
    for k, e in enumerate(eul):
        c = np.array([24 + 48 * k, 24, 24])
        M = euler_to_matrix(*e)
        # write the particle: tomo(c + M.T r) = ref(r)  (particle = ref rotated by A = M.T)
        zz, yy, xx = np.nonzero(ref)
        r = np.stack([xx, yy, zz], 1) - box // 2
        p = np.rint(r @ M + c).astype(int)
        vol[p[:, 2], p[:, 1], p[:, 0]] = 1
        rows.append(dict(rlnTomoName="t", rlnCoordinateX=c[0], rlnCoordinateY=c[1], rlnCoordinateZ=c[2],
                         rlnAngleRot=e[0], rlnAngleTilt=e[1], rlnAnglePsi=e[2]))
    write_mrc(tmp_path / "t.mrc", vol, 2.0)
    v, apix = read_mrc(tmp_path / "t.mrc")
    assert apix == 2.0 and v.shape == vol.shape
    ps = ParticleSet(pd.DataFrame(rows), ts_apix=2.0)
    res = average_particles(ps, {"t": tmp_path / "t.mrc"}, box, normalize=False)
    a = res.average
    assert a[12, 12, 16] > 0.3 and a[12, 14, 12] > 0.3      # arms recovered on +x and +y
    assert a[12, 12, 8] < 0.1 and a[12, 10, 12] < 0.1       # not mirrored


def test_tomograms_from_star(tmp_path):
    import pytest as _pytest
    from py3sta.average import tomograms_from_star

    (tmp_path / "tomos").mkdir()
    for n in ("TS_01", "TS_02"):
        write_mrc(tmp_path / "tomos" / f"{n}.mrc", np.zeros((4, 4, 4), np.float32), 1.0)
    df = pd.DataFrame({"rlnMicrographName": ["tomos/TS_01.mrc", "tomos/TS_02.mrc", "tomos/TS_01.mrc"],
                       "rlnCoordinateX": [1.0, 2, 3], "rlnCoordinateY": [1.0, 2, 3], "rlnCoordinateZ": [1.0, 2, 3]})
    from py3sta.io.star import write_star
    write_star({"particles": df}, tmp_path / "p.star")
    ps = ParticleSet.from_star(tmp_path / "p.star", ts_apix=1.0)
    m = tomograms_from_star(ps, tmp_path / "p.star")             # relative to the STAR file
    assert {k: v.name for k, v in m.items()} == {"TS_01": "TS_01.mrc", "TS_02": "TS_02.mrc"}
    m = tomograms_from_star(ps, "elsewhere.star", tomo_dir=tmp_path / "tomos")  # --tomo-dir
    assert len(m) == 2
    with _pytest.raises(FileNotFoundError, match="not found"):
        tomograms_from_star(ps, "elsewhere.star")
