import numpy as np
import pytest

from py3sta.align import (Fixed, align_maps, apply_alignment, center_of_mass, fourier_shift, load_map, mirror,
                          ncc_map, resample, rotvec_to_matrix, score_rotations, so3_grid)
from py3sta.average import box_grid, extract
from py3sta.geometry import euler_to_matrix
from py3sta.io.mrc import write_mrc
from py3sta.reference import make_mask, masked_cc

BOX, APIX = 40, 4.0


BLOBS = (((0, 0, 0), 9, 1.0), ((30, 0, 0), 6, 0.8), ((0, 22, 0), 5, 0.7), ((8, 6, 24), 4.5, 0.9),
         ((-20, -14, 6), 5, 0.6), ((14, -24, -12), 4.5, 0.7))


def chiral_object(box=BOX, apix=APIX, shift_A=(0.0, 0.0, 0.0), M=None):
    """Chiral test object: Gaussians (centre Å, sigma Å, amplitude) at non-coplanar positions,
    evaluated analytically -- optionally rotated by M (as a particle pose) and shifted, without interpolation."""
    x = box_grid(box) * apix
    v = np.zeros((box,) * 3, np.float32)
    for c, s, a in BLOBS:
        c = np.asarray(c, float) if M is None else np.asarray(M).T @ np.asarray(c, float)
        v += a * np.exp(-((x - c - np.asarray(shift_A)) ** 2).sum(-1) / (2 * s ** 2))
    return v


def posed(fixed, M, t):
    """Moving map such that the aligned map moving(c + M.T (r + t)) equals fixed(r) (trilinear)."""
    n = fixed.shape[0]
    return extract(fixed, np.full(3, n // 2, float) - np.asarray(t, float), np.asarray(M).T, n)[0]


def posed_exact(M, t, box=BOX, apix=APIX):
    """Same as posed(chiral_object(), M, t) but evaluated analytically: object(M (q - c) - t)."""
    return chiral_object(box, apix, shift_A=np.asarray(M).T @ np.asarray(t, float) * apix, M=M)


def _angle(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1)))


def test_resample_keeps_centre_and_size():
    v = chiral_object(64, 1.5)
    c0 = center_of_mass(v, 0.05) * 1.5
    out, a = resample(v, 1.5, 4.0)
    assert a == pytest.approx(4.0, rel=1e-3)
    np.testing.assert_allclose(center_of_mass(out, 0.05) * a, c0, atol=0.3)            # Å
    up, a2 = resample(out, a, 2.0)
    assert a2 == pytest.approx(2.0, rel=1e-3)
    np.testing.assert_allclose(center_of_mass(up, 0.05) * a2, c0, atol=0.3)


def test_fourier_shift_and_mirror():
    v = chiral_object()
    s = fourier_shift(v, (2, -1, 3))
    np.testing.assert_allclose(center_of_mass(s) - center_of_mass(v), (2, -1, 3), atol=0.05)
    np.testing.assert_array_equal(mirror(mirror(v))[:, :, 1:], v[:, :, 1:])      # involution (edge column lost)
    assert center_of_mass(mirror(v))[0] == pytest.approx(-center_of_mass(v)[0], abs=1e-4)


def test_ncc_map_equals_masked_cc():
    rng = np.random.default_rng(0)
    f, g = rng.normal(size=(2, 16, 16, 16)).astype(np.float32)
    g = 0.6 * np.roll(f, (1, -2, 3), (0, 1, 2)) + 0.4 * g
    m = make_mask(16, 1.0, "sphere", [5.0], soft=2.0)
    fx = Fixed.make(f, m, 4)
    nm = ncc_map(fx, g)
    for t in [(0, 0, 0), (3, -2, 1), (-1, 2, -4)]:                               # (tx, ty, tz)
        shifted = np.roll(g, (-t[2], -t[1], -t[0]), (0, 1, 2))                    # g(r + t)
        assert nm[t[2] % 16, t[1] % 16, t[0] % 16] == pytest.approx(masked_cc(f, shifted, m), abs=1e-5)
    j = np.unravel_index(np.argmax(nm), nm.shape)
    assert (j[2], j[1], j[0]) == (3, 14, 1)                                       # t = (3, -2, 1) mod 16


def test_torch_scores_match_numpy():
    pytest.importorskip("torch")
    f = chiral_object(20, 8.0)
    m = make_mask(20, 8.0, "sphere", [60.0], soft=10.0)
    mats = euler_to_matrix(*so3_grid(45)[:40].T)
    fx = Fixed.make(f, m, 3)
    a = score_rotations(fx, posed(f, mats[5], (1, 0, -1)), mats)
    b = score_rotations(fx, posed(f, mats[5], (1, 0, -1)), mats, device="cpu", batch=16)
    np.testing.assert_allclose(b[0], a[0], atol=1e-4)
    np.testing.assert_allclose(b[2], a[2], atol=1e-4)
    np.testing.assert_array_equal(b[1][a[0] > 0.5], a[1][a[0] > 0.5])
    assert np.argmax(a[0]) == 5 and tuple(a[1][5]) == (1, 0, -1)


@pytest.fixture(scope="module")
def case():
    f = chiral_object()
    M = euler_to_matrix(37.0, 71.0, -123.0)
    t = np.array([3.0, -2.0, 1.5])                                                 # voxels, fixed frame
    m = make_mask(BOX, APIX, "sphere", [70.0], soft=12.0)
    return f, M, t, m


def test_align_recovers_pose(case):
    f, M, t, m = case
    mv = posed_exact(M, t)
    assert masked_cc(posed(f, M, t), mv, m) > 0.995                              # same convention
    res = align_maps(f, mv, APIX, m, lowpass=10.0, bin_factor=1, step=30.0, max_shift_A=24.0, try_mirror=False)
    assert _angle(res.matrix, M) < 1.0, res.log
    np.testing.assert_allclose(res.shift, t, atol=0.2)
    assert res.cc > 0.99 and not res.mirrored and not res.inverted_contrast_suspected
    assert masked_cc(apply_alignment(mv, res), f, m) > 0.99


def test_align_detects_mirror_and_contrast(case):
    f, M, t, m = case
    mv = mirror(posed_exact(M, t))
    res = align_maps(f, mv, APIX, m, lowpass=10.0, bin_factor=1, step=45.0, max_shift_A=24.0, polish=False)
    assert res.mirrored and res.cc > 0.97 and res.cc_other_hand < res.cc - 0.05
    assert masked_cc(apply_alignment(mv, res), f, m) > 0.97
    inv = align_maps(f, -posed_exact(M, t), APIX, m, lowpass=10.0, bin_factor=1, step=45.0, max_shift_A=24.0, polish=False,
                     try_mirror=False)
    assert inv.inverted_contrast_suspected and inv.min_cc < -0.8


def test_load_map_resamples_and_boxes(tmp_path):
    big = chiral_object(100, 1.6)                                                  # 160 Å box at 1.6 Å
    write_mrc(tmp_path / "big.mrc", big, 1.6)
    v = load_map(tmp_path / "big.mrc", APIX, BOX)
    assert v.shape == (BOX,) * 3
    assert masked_cc(v, chiral_object(), make_mask(BOX, APIX, "sphere", [60.0], soft=10.0)) > 0.99
    off = -fourier_shift(big, (5, 0, -4))                                        # off-centre, dark density
    write_mrc(tmp_path / "off.mrc", off, 1.6)
    c = load_map(tmp_path / "off.mrc", APIX, BOX, center=True, invert=True)
    np.testing.assert_allclose(center_of_mass(c, 0.05), np.zeros(3), atol=0.3)      # inverted, then centred


def test_rotvec():
    R = rotvec_to_matrix(np.array([0, 0, np.pi / 2]))
    np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(rotvec_to_matrix(np.zeros(3)), np.eye(3))
