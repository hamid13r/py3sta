import numpy as np
import pandas as pd
import pytest

from py3.geometry import euler_to_matrix
from py3.io.mrc import write_mrc
from py3.io.particles import ParticleSet
from py3.wedge import TiltInfo, rfft_grid, wedge_weights

XML = """<?xml version="1.0" encoding="utf-16"?>
<TiltSeries AreAnglesInverted="True">
\t<Angles>-30
0
30
60</Angles>
\t<Dose>9
0
3
6</Dose>
\t<UseTilt>True
True
True
False</UseTilt>
</TiltSeries>"""


def test_warp_xml(tmp_path):
    p = tmp_path / "ts.mrc.xml"
    p.write_bytes(XML.encode("utf-16"))
    t = TiltInfo.from_warp_xml(p)
    np.testing.assert_allclose(t.angles, [-30, 0, 30])  # UseTilt=False dropped
    np.testing.assert_allclose(t.dose, [9, 0, 3])
    assert t.sign == -1 and t.range == (-30, 30)


def test_tlt(tmp_path):
    p = tmp_path / "a.tlt"
    p.write_text("TILT SERIES T\nIMAGE 0 FILE T ORIGIN [1 1] TILT ANGLE -51.00 ROTATION 0\n"
                 "IMAGE 1 FILE T ORIGIN [1 1] TILT ANGLE 48.00 ROTATION 0\n")
    assert TiltInfo.from_tlt(p).range == (-51, 48)


def test_wedge_geometry():
    t = TiltInfo.from_range(-40, 60, 2)
    w = wedge_weights(32, t, soft_deg=0)
    kx, ky, kz = rfft_grid(32)
    beta = np.degrees(np.arctan2(kz, kx))
    sel = (kx > 0.05)
    assert w[sel & (beta > -38) & (beta < 58)].min() == 1      # inside sampled
    assert w[sel & ((beta < -42) | (beta > 62))].max() == 0     # missing wedge
    assert w[np.abs(kx) + np.abs(kz) == 0].min() == 1           # tilt axis (ky) always sampled
    # rotating the particle 90 deg about y rotates the wedge accordingly: k_ref = M k_tomo
    M = euler_to_matrix(0, 90, 0)
    wr = wedge_weights(32, t, M, soft_deg=0)
    kt = np.einsum("ji,j...->i...", M, np.stack([kx, ky, kz]))
    bt = np.degrees(np.arctan2(np.where(kt[0] < 0, -kt[2], kt[2]), np.abs(kt[0])))
    ok = np.abs(kt[0]) + np.abs(kt[2]) > 1e-6
    np.testing.assert_array_equal(wr[ok], ((bt >= -40) & (bt <= 60))[ok].astype(np.float32))


def test_dose_weights_decrease():
    t = TiltInfo(np.array([-30.0, 0, 30]), np.array([60.0, 0, 30]), 1.0)
    w = wedge_weights(32, t, apix=5.0, dose_weight=True, soft_deg=0)
    kx, ky, kz = rfft_grid(32)
    b = np.degrees(np.arctan2(kz, kx))
    r = np.sqrt(kx**2 + ky**2 + kz**2)
    shell = (r > 0.3) & (r < 0.4) & (kx > 0)
    assert w[shell & (np.abs(b) < 5)].mean() > w[shell & (b > 25)].mean() > w[shell & (b < -25)].mean()


def test_wedge_average_recovers_object(tmp_path):
    """Wedge-weighted average of randomly oriented copies matches the object."""
    from py3.average import average_particles

    box = 20
    z, y, x = np.mgrid[:box, :box, :box] - box // 2
    obj = (np.exp(-((x - 3) ** 2 + y ** 2 + z ** 2) / 4) + 0.6 * np.exp(-(x ** 2 + (y - 4) ** 2 + (z + 2) ** 2) / 3)).astype(np.float32)
    rng = np.random.default_rng(0)
    n = 40
    vol = np.zeros((box, box, box * n), np.float32)
    rows = []
    t = TiltInfo.from_range(-60, 60, 2)
    from py3.average import box_grid, _trilinear
    g = box_grid(box).reshape(-1, 3)
    for i in range(n):
        e = (rng.uniform(-180, 180), np.degrees(np.arccos(rng.uniform(-1, 1))), rng.uniform(-180, 180))
        M = euler_to_matrix(*e)
        # particle in tomo frame: p(r) = obj(M r), then apply the tomogram-frame wedge
        part, _ = _trilinear(np.pad(obj, 1), g @ M.T + box // 2 + 1)
        part = np.fft.irfftn(np.fft.rfftn(part.reshape((box,) * 3)) * wedge_weights(box, t, soft_deg=0), s=(box,) * 3)
        vol[:, :, i * box:(i + 1) * box] = part
        rows.append(dict(rlnTomoName="t", rlnCoordinateX=i * box + box // 2, rlnCoordinateY=box // 2,
                         rlnCoordinateZ=box // 2, rlnAngleRot=e[0], rlnAngleTilt=e[1], rlnAnglePsi=e[2]))
    write_mrc(tmp_path / "t.mrc", vol, 1.0)
    ps = ParticleSet(pd.DataFrame(rows), 1.0)
    kw = dict(normalize=False, min_inside=0.0)
    plain = average_particles(ps, {"t": tmp_path / "t.mrc"}, box, **kw).average
    wedge = average_particles(ps, {"t": tmp_path / "t.mrc"}, box, tilts={"t": t}, **kw).average

    def cc(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

    assert cc(wedge, obj) > 0.9
    assert cc(wedge, obj) >= cc(plain, obj) - 0.01


def test_torch_matches_numpy(tmp_path):
    torch = pytest.importorskip("torch")
    from py3 import average as A
    from py3 import backend_torch as T

    rng = np.random.default_rng(2)
    vol = rng.normal(size=(60, 70, 80)).astype(np.float32)
    c = rng.uniform(15, 45, size=(4, 3))
    M = euler_to_matrix(*rng.uniform(0, 90, size=(3, 4)))
    sub, _ = T.extract_batch(vol, c, M, 16, torch.device("cpu"))
    for i in range(4):
        np.testing.assert_allclose(sub[i].numpy(), A.extract(vol, c[i], M[i], 16)[0], atol=1e-4)
    t = TiltInfo(np.array([-50.0, 0, 40]), np.array([5.0, 0, 10]), -1.0)
    wt = T.wedge_weights_batch(16, t, M, 4.0, True, device="cpu").numpy()
    for i in range(4):
        np.testing.assert_allclose(wt[i], wedge_weights(16, t, M[i], 4.0, True), atol=1e-5)
