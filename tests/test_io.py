import numpy as np
import pandas as pd
import pytest

from py3sta.geometry import euler_to_matrix, frame_from_normal, is_rotation, matrix_to_euler, relion_A
from py3sta.io.dipole import dipoles_to_particles
from py3sta.io.particles import ParticleSet
from py3sta.io.star import read_star
from py3sta.io.trf import matrices, read_trf

# three real lines from the i3 protocol docs (getting_started.md)
TRF_DOC = """\
NAD054_1_lam12_ts_004.mrc_0000000 46 421 72 0.0 0.0 0.0 0.25045375749633164 -0.9450187992901435 0.2102674115125402 0.09737071909722955 -0.19149986131638969 -0.976650780053081 0.9632195276355524 0.2650797466877067 0.04405530022056553
NAD054_1_lam12_ts_004.mrc_0000001 449 474 69 0.0 0.0 0.0 0.42774992428180214 0.8057929694911962 -0.40954571490307706 -0.34729320510027895 -0.2717994224956718 -0.8975034839053242 -0.8345162862209987 0.5261394912469308 0.16358424062950772
NAD054_1_lam12_ts_004.mrc_0000002 495 588 80 0.0 0.0 0.0 0.4589448088207747 -0.877093460401622 -0.14169235751151515 -0.7815058323258316 -0.3226645287341817 -0.5339814940027957 0.42263257860365183 0.35580143847019063 -0.8335388652518677
"""

rng = np.random.default_rng(0)


def random_eulers(n):
    return np.column_stack([rng.uniform(-180, 180, n), np.degrees(np.arccos(rng.uniform(-1, 1, n))),
                            rng.uniform(-180, 180, n)])


# ---------------------------------------------------------------- geometry
def test_euler_roundtrip():
    e = random_eulers(500)
    m = euler_to_matrix(*e.T)
    assert is_rotation(m).all()
    np.testing.assert_allclose(euler_to_matrix(*matrix_to_euler(m).T), m, atol=1e-9)


def test_gimbal_lock():
    for tilt in (0.0, 180.0):
        m = euler_to_matrix(30.0, tilt, 50.0)
        np.testing.assert_allclose(euler_to_matrix(*matrix_to_euler(m)), m, atol=1e-9)


def test_matches_relion_source_convention():
    # RELION's Euler_angles2matrix A is the transpose of our M (tomo -> reference)
    for rot, tilt, psi in random_eulers(50):
        np.testing.assert_allclose(relion_A(rot, tilt, psi).T, euler_to_matrix(rot, tilt, psi), atol=1e-12)


def test_matches_scipy_zyz_if_available():
    R = pytest.importorskip("scipy.spatial.transform").Rotation
    e = random_eulers(200)
    np.testing.assert_allclose(R.from_euler("ZYZ", e, degrees=True).as_matrix(), euler_to_matrix(*e.T), atol=1e-12)
    m = euler_to_matrix(*e.T)
    np.testing.assert_allclose(R.from_matrix(m).as_euler("ZYZ", degrees=True), matrix_to_euler(m), atol=1e-6)


def test_frame_from_normal():
    n = rng.normal(size=(200, 3))
    m = frame_from_normal(n, rng.uniform(0, 360, 200))
    assert is_rotation(m).all()
    nz = m @ (n / np.linalg.norm(n, axis=1, keepdims=True))[..., None]
    np.testing.assert_allclose(nz[..., 0], np.tile([0, 0, 1.0], (200, 1)), atol=1e-12)
    # normal parallel to z must not crash
    assert is_rotation(frame_from_normal([0, 0, 5.0]))


# --------------------------------------------------------------- trf / star
def _doc_trf(tmp_path):
    p = tmp_path / "doc.trf"
    p.write_text(TRF_DOC)
    return p


def test_read_doc_trf(tmp_path):
    t = read_trf(_doc_trf(tmp_path))
    assert len(t) == 3 and is_rotation(matrices(t), tol=1e-6).all()


def test_legacy_trf2star_equivalence(tmp_path):
    """Same angles as the original trf2star.py (scipy as_euler('ZYZ'))."""
    R = pytest.importorskip("scipy.spatial.transform").Rotation
    t = read_trf(_doc_trf(tmp_path))
    legacy = R.from_matrix(matrices(t)).as_euler("ZYZ", degrees=True)
    ps = ParticleSet.from_trf(_doc_trf(tmp_path), tomo_apix=13.3)
    np.testing.assert_allclose(ps.df[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(), legacy, atol=1e-6)


def test_trf_star_trf_roundtrip(tmp_path):
    t = read_trf(_doc_trf(tmp_path))
    t.loc[:, ["dx", "dy", "dz"]] = [[0.3, -0.2, 0.1], [0.0, 0.4, -0.45], [-0.1, 0.0, 0.2]]
    from py3sta.io.trf import write_trf
    write_trf(t, tmp_path / "in.trf")
    ps = ParticleSet.from_trf(tmp_path / "in.trf", tomo_apix=13.3, ts_apix=1.33)
    assert set(ps.df.rlnTomoName) == {"NAD054_1_lam12_ts_004"}
    ps.to_relion4(tmp_path / "p.star")
    ps2 = ParticleSet.from_star(tmp_path / "p.star")
    assert ps2.ts_apix == pytest.approx(1.33)
    out = ps2.to_trf(tmp_path / "trf", tomo_apix=13.3)
    t2 = read_trf(out["NAD054_1_lam12_ts_004"])
    true1 = t[["x", "y", "z"]].to_numpy() + t[["dx", "dy", "dz"]].to_numpy()
    true2 = t2[["x", "y", "z"]].to_numpy() + t2[["dx", "dy", "dz"]].to_numpy()
    np.testing.assert_allclose(true2, true1, atol=1e-4)
    np.testing.assert_allclose(matrices(t2), matrices(t), atol=1e-6)


def test_relion4_layout(tmp_path):
    ps = ParticleSet.from_trf(_doc_trf(tmp_path), tomo_apix=13.3, ts_apix=1.33)
    ps.to_relion4(tmp_path / "p.star")
    b = read_star(tmp_path / "p.star")
    assert set(b) == {"optics", "particles"}
    assert b["optics"]["rlnTomoTiltSeriesPixelSize"].iloc[0] == pytest.approx(1.33)
    p = b["particles"]
    for c in ["rlnTomoName", "rlnCoordinateX", "rlnOriginXAngst", "rlnAngleRot", "rlnMicrographName"]:
        assert c in p
    np.testing.assert_allclose(p["rlnCoordinateX"], [460, 4490, 4950])  # 10x binning
    assert p["rlnMicrographName"].iloc[0] == "NAD054_1_lam12_ts_004.tomostar"


def test_warp_export_folds_origins(tmp_path):
    df = pd.DataFrame({"rlnTomoName": ["TS_01"], "rlnCoordinateX": [100.0], "rlnCoordinateY": [50.0],
                       "rlnCoordinateZ": [20.0], "rlnOriginXAngst": [5.0], "rlnOriginYAngst": [-2.5],
                       "rlnOriginZAngst": [0.0]})
    ps = ParticleSet(df, ts_apix=2.5)
    assert ps.to_warp(tmp_path / "w.star") == 2.5
    b = read_star(tmp_path / "w.star")
    w = b[""]
    assert list(w.columns[:7]) == ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ",
                                   "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnMicrographName"]
    np.testing.assert_allclose(w[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy()[0], [98, 51, 20])
    assert w["rlnMicrographName"].iloc[0] == "TS_01.tomostar"
    # Warp-style file reads back
    assert len(ParticleSet.from_star(tmp_path / "w.star", ts_apix=2.5)) == 1


def test_duplicate_positions_rejected(tmp_path):
    df = pd.DataFrame({"rlnTomoName": ["a", "a"], "rlnCoordinateX": [1.0, 1.1],
                       "rlnCoordinateY": [2.0, 2.0], "rlnCoordinateZ": [3.0, 3.0]})
    with pytest.raises(ValueError, match="duplicates"):
        ParticleSet(df, 1.0).to_trf(tmp_path, 1.0)


# ------------------------------------------------------------------ dipoles
def test_dipoles(tmp_path):
    p = tmp_path / "TS_01.mrc_13.30Apx.pos"
    p.write_text("10 10 10\n10 10 20\n\n30.5 40  50\n40.5 40 50\n")
    ps = dipoles_to_particles([p], pick_apix=13.3, ts_apix=1.33)
    assert list(ps.df.rlnTomoName) == ["TS_01", "TS_01"]
    np.testing.assert_allclose(ps.df[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(),
                               [[100, 100, 150], [355, 400, 500]])  # sub-voxel kept, 10x scale
    m = ps.matrices
    normals = np.array([[0, 0, 1.0], [1.0, 0, 0]])
    np.testing.assert_allclose(np.einsum("nij,nj->ni", m, normals), [[0, 0, 1]] * 2, atol=1e-12)
    np.testing.assert_allclose(ps.df.rlnAngleTilt, ps.df.rlnAngleTiltPrior)


def test_dipoles_match_legacy_pos2i3(tmp_path):
    """Non-random orientation equals pos2i3.py's scipy align_vectors result."""
    R = pytest.importorskip("scipy.spatial.transform").Rotation
    a = rng.normal(size=(40, 3)) * 50 + 200
    p = tmp_path / "t.pos"
    p.write_text("\n".join(" ".join(f"{v:.3f}" for v in r) for r in a))
    ps = dipoles_to_particles([p], 1.0)
    for k in range(20):
        v = a[2 * k + 1] - a[2 * k]
        nv = np.cross(v / np.linalg.norm(v), [0, 0, 1]) / np.linalg.norm(v)
        rot, _ = R.align_vectors([[0, 0, 1], [0, 1, 0], [0, -1, 0]], [v / np.linalg.norm(v), nv, -nv])
        np.testing.assert_allclose(ps.matrices[k], rot.as_matrix(), atol=1e-3)  # scipy versions differ ~1e-5; picks need no more


def test_odd_pos_rejected(tmp_path):
    p = tmp_path / "bad.pos"
    p.write_text("1 2 3\n4 5 6\n7 8 9\n")
    with pytest.raises(ValueError, match="odd"):
        dipoles_to_particles([p], 1.0)


def test_mrc_axis_order(tmp_path):
    """MAPC/R/S = (2, 1, 3) (stored [z][x][y], as some EMDB maps) is read back as [z, y, x]."""
    from py3sta.io.mrc import header_bytes, read_header, read_mrc

    vol = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)      # [z, y, x]
    stored = vol.transpose(0, 2, 1)                                      # [z][x][y]: columns = y
    hb = bytearray(header_bytes(stored.shape, np.float32, 2.0))
    i = np.frombuffer(bytes(hb), "<i4").copy()
    i[16:19] = (2, 1, 3)
    i[7:10] = (6, 5, 4)                                                  # MX, MY, MZ follow x, y, z
    f = i.view("<f4")
    f[10:13] = (12.0, 10.0, 8.0)
    p = tmp_path / "p.mrc"
    p.write_bytes(i.tobytes() + np.ascontiguousarray(stored).tobytes())
    assert read_header(p)["shape"] == (4, 5, 6)
    for mmap in (True, False):
        out, apix = read_mrc(p, mmap=mmap)
        np.testing.assert_array_equal(out, vol)
        assert apix == 2.0
