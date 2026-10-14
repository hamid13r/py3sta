import numpy as np
import pandas as pd
import pytest

from py3sta.geometry import euler_to_matrix
from py3sta.io.mrc import write_mrc
from py3sta.io.particles import ParticleSet
from py3sta.wedge import TiltInfo, rfft_grid, wedge_weights

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
    from py3sta.average import average_particles

    box = 20
    z, y, x = np.mgrid[:box, :box, :box] - box // 2
    obj = (np.exp(-((x - 3) ** 2 + y ** 2 + z ** 2) / 4) + 0.6 * np.exp(-(x ** 2 + (y - 4) ** 2 + (z + 2) ** 2) / 3)).astype(np.float32)
    rng = np.random.default_rng(0)
    n = 40
    vol = np.zeros((box, box, box * n), np.float32)
    rows = []
    t = TiltInfo.from_range(-60, 60, 2)
    from py3sta.average import box_grid, _trilinear
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
    from py3sta import average as A
    from py3sta import backend_torch as T

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


XML_CTF = """<?xml version="1.0" encoding="utf-16"?>
<TiltSeries AreAnglesInverted="True">
<Angles>-30
0
30</Angles>
<UseTilt>True
True
False</UseTilt>
<CTF><Param Name="Voltage" Value="300.0" /><Param Name="Cs" Value="2.7" /><Param Name="Amplitude" Value="0.07" />
<Param Name="Defocus" Value="5.0" /><Param Name="PhaseShift" Value="0" /></CTF>
<GridCTF Width="1" Height="1" Depth="3"><Node X="0" Y="0" Z="1" Value="4.0" /><Node X="0" Y="0" Z="0" Value="3.0" />
<Node X="0" Y="0" Z="2" Value="5.0" /></GridCTF>
</TiltSeries>"""


def test_warp_xml_ctf(tmp_path):
    p = tmp_path / "ts.xml"
    p.write_bytes(XML_CTF.encode("utf-16"))
    t = TiltInfo.from_warp_xml(p)
    np.testing.assert_allclose(t.ctf.defocus_um, [3.0, 4.0])   # sorted by Z, unused tilt dropped
    assert t.ctf.voltage_kv == 300 and t.ctf.amplitude == 0.07


def test_ctf_zeros_and_defocus_geometry():
    from py3sta.wedge import CTFParams, ctf_values, electron_wavelength, particle_defocus

    assert abs(electron_wavelength(300) - 0.019687) < 1e-5
    ctf = CTFParams(np.array([6.5]), cs_mm=0.0, amplitude=0.0)
    lam = electron_wavelength(300)
    k1 = np.sqrt(1 / (lam * 6.5e4))                         # first zero for Cs=0, A=0
    assert abs(ctf_values(np.array([k1]), 6.5e4, ctf)[0]) < 1e-6
    assert ctf_values(np.array([0.5 * k1]), 6.5e4, ctf)[0] < 0   # convention: negative before first zero
    t = TiltInfo(np.array([0.0, 30.0]), None, 1.0, ctf=CTFParams(np.array([5.0, 5.0])))
    d = particle_defocus(t, (1000.0, 0, 500.0), height_sign=1)
    # tilt 0: height = z; tilt 30: height = -x sin30 + z cos30
    np.testing.assert_allclose(d, [5e4 + 500, 5e4 - 500 + 500 * np.cos(np.radians(30))], atol=1e-6)


def test_ctf_modes():
    from py3sta.wedge import CTFParams

    t = TiltInfo(np.array([-40.0, 0, 40]), None, 1.0, ctf=CTFParams(np.array([4.0, 4.0, 4.0])))
    dfc = np.full(3, 4e4)
    w = wedge_weights(32, t, apix=5.0, defocus_A=dfc, ctf_mode="wiener", soft_deg=0)
    f = wedge_weights(32, t, apix=5.0, defocus_A=dfc, ctf_mode="flip", soft_deg=0)
    a = wedge_weights(32, t, apix=5.0, defocus_A=dfc, ctf_mode="amplitude", soft_deg=0)
    plain = wedge_weights(32, t, soft_deg=0)
    inside = plain > 0
    np.testing.assert_allclose(np.abs(w), a, atol=1e-6)
    np.testing.assert_allclose(np.abs(f[inside]), 1)
    np.testing.assert_allclose(np.sign(w[inside & (np.abs(w) > 1e-3)]), f[inside & (np.abs(w) > 1e-3)])
    assert w[0, 0, 1] > 0                                    # low frequency keeps its contrast


def test_torch_ctf_matches_numpy():
    torch = pytest.importorskip("torch")
    from py3sta import backend_torch as T
    from py3sta.wedge import CTFParams, particle_defocus

    rng = np.random.default_rng(3)
    M = euler_to_matrix(*rng.uniform(0, 90, size=(3, 3)))
    t = TiltInfo(np.array([-50.0, 0, 40]), np.array([5.0, 0, 10]), -1.0, ctf=CTFParams(np.array([5.0, 6.0, 7.0])))
    dfc = np.stack([particle_defocus(t, rng.uniform(-500, 500, 3)) for _ in range(3)])
    for mode in ("flip", "wiener", "amplitude"):
        wt = T.wedge_weights_batch(16, t, M, 4.0, True, device="cpu", defocus_A=dfc, ctf_mode=mode).numpy()
        for i in range(3):
            np.testing.assert_allclose(wt[i], wedge_weights(16, t, M[i], 4.0, True, defocus_A=dfc[i], ctf_mode=mode),
                                       atol=1e-4)
