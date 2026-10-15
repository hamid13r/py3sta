import numpy as np
import pandas as pd
import pytest

from py3sta.align import mirror
from py3sta.geometry import euler_to_matrix
from py3sta.io.particles import ParticleSet
from py3sta.mra import MRASettings, align_particles, cone_grid, filter_crop, prepare, score_numpy
from py3sta.poses import apply_delta, compose, invert
from py3sta.reference import make_mask
from py3sta.stack import StackLocation
from py3sta.wedge import TiltInfo, wedge_weights

from test_align import APIX, BOX, chiral_object

TOMO, MARGIN = 96, 4
TILT = TiltInfo.from_range(-50.0, 50.0)


def _angle(A, B):
    return np.degrees(np.arccos(np.clip((np.einsum("...ij,...ij->...", A, B) - 1) / 2, -1, 1)))


def test_cone_grid():
    g = cone_grid(10, 5, 10, 5)
    np.testing.assert_allclose(g.mats[0], np.eye(3))
    assert np.allclose(np.einsum("kij,kil->kjl", g.mats, g.mats), np.eye(3))       # rotations
    axis_tilt = np.degrees(np.arccos(np.clip(g.mats[:, 2, 2], -1, 1)))             # angle of D.T e_z to e_z
    np.testing.assert_allclose(axis_tilt, g.theta, atol=1e-6)
    assert axis_tilt.max() == pytest.approx(10) and len(g.mats) == (1 + 6 + 13) * 5
    z = cone_grid(0, 0, 30, 10)                                                     # about z only
    assert len(z.mats) == 7 and np.allclose(z.mats[:, 2], [0, 0, 1])
    assert len(cone_grid(0, 0, 0, 0).mats) == 1
    assert g.at_edge(len(g.mats) - 1) and not g.at_edge(0)


class FakeStack:
    """Minimal ParticleStack: crops of box + 2 margin around the rounded particle centres."""

    def __init__(self, tomo, ps):
        self.apix, self.margin = APIX, MARGIN
        size = BOX + 2 * MARGIN
        c = ps.centers_px * ps.ts_apix / APIX
        self.lo = np.round(c).astype(int) - size // 2
        self.crops = np.stack([tomo[l[2]:l[2] + size, l[1]:l[1] + size, l[0]:l[0] + size] for l in self.lo])
        self.shape = tomo.shape

    def locate(self, ps, shift_limit=0):
        return StackLocation(np.arange(len(ps)), self.lo, {"TS_01": self.shape})

    def data(self, mode="r"):
        return self.crops


def _scene(n=6, seed=0, wedge=True, classes=None, max_err=8.0, max_shift=1.5):
    """Analytic tomogram with the chiral object (or its mirror for class 2) at true poses; the STAR poses
    are off by a known per-particle delta. Returns (stack, given particles, true particles)."""
    rng = np.random.default_rng(seed)
    pos = np.array([[x, y, z] for x in (24, 72) for y in (24, 72) for z in (24, 72)], float)[:n]
    pos = pos + rng.uniform(-0.5, 0.5, pos.shape)
    e = np.column_stack([rng.uniform(-180, 180, n), rng.uniform(10, 170, n), rng.uniform(-180, 180, n)])
    df = pd.DataFrame({"rlnTomoName": "TS_01", "rlnCoordinateX": pos[:, 0], "rlnCoordinateY": pos[:, 1],
                       "rlnCoordinateZ": pos[:, 2], "rlnAngleRot": e[:, 0], "rlnAngleTilt": e[:, 1],
                       "rlnAnglePsi": e[:, 2]})
    true = ParticleSet(df, APIX)
    tomo = np.zeros((TOMO,) * 3, np.float32)
    for i, (M, c) in enumerate(zip(true.matrices, true.centers_px)):
        obj = chiral_object(TOMO, APIX, shift_A=(c - TOMO // 2) * APIX, M=M)
        if classes is not None and classes[i] == 2:                     # mirrored object about the particle x
            obj = chiral_object(TOMO, APIX, shift_A=(c - TOMO // 2) * APIX, M=np.diag([-1.0, 1, 1]) @ M)
        tomo += obj
    if wedge:
        w = wedge_weights(TOMO, TILT, soft_deg=0)
        tomo = np.fft.irfftn(np.fft.rfftn(tomo) * w, s=tomo.shape).astype(np.float32)
    tomo += rng.normal(scale=0.02, size=tomo.shape).astype(np.float32)
    R = euler_to_matrix(*rng.uniform(-max_err, max_err, (3, n)))
    t = rng.uniform(-max_shift, max_shift, (n, 3))
    given = apply_delta(true, *invert(R, t), APIX)
    return FakeStack(tomo, true), given, true


MASK = make_mask(BOX, APIX, "sphere", [62.0], soft=12.0)


def _errors(given, res, true):
    M, c = compose(given.matrices, given.centers_px, res.R, res.t)
    return _angle(M, true.matrices), np.abs(c - true.centers_px).max(1)


def test_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    stack, given, _ = _scene(n=1)
    for s in (MRASettings(lowpass=10.0, max_shift=3), MRASettings(lowpass=12.0, max_shift=4, bin=2)):
        pr = prepare(chiral_object()[None], MASK, APIX, s)
        crop = filter_crop(stack.crops[0], pr)
        mats = cone_grid(6, 3, 6, 3).mats[:12] @ given.matrices[0]
        c = given.centers_px[0]
        from py3sta.mra import _TorchScorer

        sc = _TorchScorer(pr, torch.device("cpu"), batch=5)
        for tilt in (None, TILT):
            a_cc, a_t, _ = score_numpy(pr, crop, stack.lo[0], stack.shape, c, mats, 0, tilt)
            b_cc, b_t, _ = sc.score(torch.as_tensor(crop[None]), np.zeros(len(mats), int), np.repeat(stack.lo[:1], 12, 0),
                                    stack.shape, np.repeat(c[None], 12, 0), mats, 0, tilt)
            np.testing.assert_allclose(b_cc, a_cc, atol=2e-4)
            np.testing.assert_array_equal(b_t, a_t)


@pytest.fixture(scope="module")
def device():
    torch = pytest.importorskip("torch")
    return torch.device("cpu")


SEARCH = MRASettings(levels=[[10.0, 5.0, 10.0, 5.0], [4.0, 2.0, 4.0, 2.0], [1.0, 0.5, 1.0, 0.5]], max_shift=3,
                      lowpass=10.0)


def test_recovers_known_deltas(device):
    stack, given, true = _scene(n=6, wedge=True)
    res = align_particles(given, stack, chiral_object()[None], MASK, APIX, SEARCH, tilts={"TS_01": TILT},
                          device=device)
    ang, sh = _errors(given, res, true)
    assert np.median(ang) < 1.0 and ang.max() < 2.0, ang
    assert sh.max() < 0.3, sh
    assert (res.best_ref == 1).all() and (res.cc > 0.8).all() and not res.at_edge.any()


def test_numpy_driver_matches_torch(device):
    stack, given, true = _scene(n=1, wedge=True, seed=3)
    s = MRASettings(levels=[[4.0, 2.0, 4.0, 2.0]], max_shift=2, lowpass=12.0, bin=2)
    a = align_particles(given, stack, chiral_object()[None], MASK, APIX, s, tilts={"TS_01": TILT})
    b = align_particles(given, stack, chiral_object()[None], MASK, APIX, s, tilts={"TS_01": TILT}, device=device)
    np.testing.assert_allclose(a.R, b.R, atol=1e-6)
    np.testing.assert_allclose(a.t, b.t, atol=1e-3)
    np.testing.assert_allclose(a.cc, b.cc, atol=2e-4)


def test_wedge_constrained_beats_plain(device):
    """Strongly wedged data: the constrained correlation finds the true pose better than plain CC."""
    stack, given, true = _scene(n=8, wedge=True, seed=5)
    errs = {}
    for wedge in (True, False):
        s = MRASettings(levels=SEARCH.levels, max_shift=3, lowpass=10.0, wedge=wedge)
        res = align_particles(given, stack, chiral_object()[None], MASK, APIX, s, tilts={"TS_01": TILT}, device=device)
        errs[wedge] = _errors(given, res, true)[0]
    assert errs[True].mean() < errs[False].mean(), errs
    assert np.median(errs[True]) < 1.0


def test_multi_reference_best_cc(device):
    classes = np.array([1, 2, 1, 2, 1, 2])
    stack, given, true = _scene(n=6, wedge=False, classes=classes, seed=7)
    refs = np.stack([chiral_object(), mirror(chiral_object())])
    s = MRASettings(levels=SEARCH.levels, max_shift=3, lowpass=10.0, bin=2)
    res = align_particles(given, stack, refs, MASK, APIX, s, device=device)
    np.testing.assert_array_equal(res.best_ref, classes)
    assert (res.cc_ref.max(1) - res.cc_ref.min(1) > 0.1).all()


def test_edge_flag(device):
    stack, given, true = _scene(n=2, wedge=False, max_err=20.0, seed=9)
    s = MRASettings(levels=[[4.0, 2.0, 4.0, 2.0]], max_shift=3, lowpass=12.0, bin=2)
    res = align_particles(given, stack, chiral_object()[None], MASK, APIX, s, device=device)
    assert res.at_edge.all()
