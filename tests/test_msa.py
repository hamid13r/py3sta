import numpy as np
import pandas as pd
import pytest

from py3sta.geometry import euler_to_matrix
from py3sta.io.particles import ParticleSet
from py3sta.msa import (MSASettings, build_stack, correlation_ratio, eigenvolumes, factor_diagnostics,
                        multiple_r, pca, wedge_features)
from py3sta.reference import make_mask
from py3sta.stack import StackLocation
from py3sta.wedge import TiltInfo, wedge_weights

from test_align import APIX, BOX, chiral_object

MARGIN = 4
SIZE = BOX + 2 * MARGIN
TILT = TiltInfo.from_range(-50.0, 50.0)
EXTRA = ((-10.0, 20.0, -16.0), 5.0, 1.0)             # state 2: one more Gaussian (centre Å, sigma Å, amplitude)
MASK = make_mask(BOX, APIX, "sphere", [62.0], soft=12.0)


class CropStack:
    """One crop per particle, each its own little tomogram (lo = 0), particle at the crop centre."""

    def __init__(self, crops):
        self.apix, self.margin, self.crops = APIX, MARGIN, crops

    def locate(self, ps, shift_limit=0):
        return StackLocation(np.arange(len(ps)), np.zeros((len(ps), 3), int), {"TS_01": (SIZE,) * 3})

    def data(self, mode="r"):
        return self.crops


def two_states(n=80, seed=0, noise=0.3, wedge=True):
    """Chiral object (state 1) or chiral object + one blob (state 2) at random poses, wedge applied in
    the tomogram frame, Gaussian noise. Returns (stack, particles, states)."""
    rng = np.random.default_rng(seed)
    e = np.column_stack([rng.uniform(-180, 180, n), np.degrees(np.arccos(rng.uniform(-1, 1, n))),
                         rng.uniform(-180, 180, n)])
    states = np.tile([1, 2], n // 2 + 1)[:n]
    rng.shuffle(states)
    c = SIZE // 2
    df = pd.DataFrame({"rlnTomoName": "TS_01", "rlnCoordinateX": float(c), "rlnCoordinateY": float(c),
                       "rlnCoordinateZ": float(c), "rlnAngleRot": e[:, 0], "rlnAngleTilt": e[:, 1],
                       "rlnAnglePsi": e[:, 2]})
    ps = ParticleSet(df, APIX)
    from py3sta.average import box_grid

    x = box_grid(SIZE) * APIX
    w = wedge_weights(SIZE, TILT, soft_deg=0)
    crops = []
    for M, s in zip(ps.matrices, states):
        v = chiral_object(SIZE, APIX, M=M)
        if s == 2:
            p, sig, a = EXTRA
            v = v + a * np.exp(-((x - M.T @ np.asarray(p)) ** 2).sum(-1) / (2 * sig ** 2))
        if wedge:
            v = np.fft.irfftn(np.fft.rfftn(v) * w, s=v.shape, axes=(0, 1, 2))
        crops.append(v + rng.normal(scale=noise, size=v.shape))
    return CropStack(np.array(crops, np.float32)), ps, states


@pytest.fixture(scope="module")
def scene():
    return two_states()


def _accuracy(score, states):
    a = np.mean((score > np.median(score)) == (states == 2))
    return max(a, 1 - a)


S16 = dict(bin=2, lowpass=16.0)


def test_wmd_separates_states_plain_shows_wedge(scene):
    stack, ps, states = scene
    tilts = {"TS_01": TILT}
    res = {}
    for name, s in (("wmd", MSASettings(mode="wmd", **S16)), ("plain", MSASettings(mode="plain", wedge=False, **S16)),
                    ("plain_wedge", MSASettings(mode="plain", wedge=True, **S16))):
        st = build_stack(ps, stack, MASK, APIX, s, tilts=tilts)
        res[name] = pca(st.X, 5)
    F = wedge_features(ps.matrices)
    diag = {k: factor_diagnostics(r.scores, ps.matrices) for k, r in res.items()}
    # wmd: the state is the leading factor, not the wedge
    assert _accuracy(res["wmd"].scores[:, 0], states) > 0.95
    assert diag["wmd"]["wedge_r"][0] < 0.6
    # plain: the leading factor follows the wedge orientation and misses the state
    for k in ("plain", "plain_wedge"):
        assert diag[k]["wedge_r"][0] > 0.9, diag[k]
        assert _accuracy(res[k].scores[:, 0], states) < 0.75
    assert F.shape == (len(ps), 12)


def test_torch_matches_numpy(scene):
    torch = pytest.importorskip("torch")
    stack, ps, _ = scene
    sub = ParticleSet(ps.df.iloc[:6].reset_index(drop=True), APIX)
    for s in (MSASettings(mode="wmd", **S16), MSASettings(mode="plain", wedge=True, bin=1, lowpass=10.0),
              MSASettings(mode="plain", wedge=False, **S16)):
        a = build_stack(sub, stack, MASK, APIX, s, tilts={"TS_01": TILT})
        b = build_stack(sub, stack, MASK, APIX, s, tilts={"TS_01": TILT}, device=torch.device("cpu"))
        scale = np.abs(a.X).max()
        np.testing.assert_allclose(b.X, a.X, atol=2e-4 * scale)
        np.testing.assert_allclose(b.average, a.average, atol=2e-4 * np.abs(a.average).max())


def test_pca_numpy_torch_and_exact():
    rng = np.random.default_rng(1)
    n, v, k = 300, 200, 4
    basis = np.linalg.qr(rng.normal(size=(v, k)))[0].T
    X = (rng.normal(size=(n, k)) * [10, 6, 4, 2]) @ basis + rng.normal(scale=0.3, size=(n, v)) + 5.0
    X = X.astype(np.float32)
    a = pca(X, k, row_chunk=64)
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc.astype(np.float64), full_matrices=False)
    for j in range(k):
        assert abs(abs(a.vectors[j] @ Vt[j]) - 1) < 1e-4
    np.testing.assert_allclose(a.eigenvalues, s[:k] ** 2 / (n - 1), rtol=1e-4)
    np.testing.assert_allclose(a.var_fraction.sum(), (s[:k] ** 2).sum() / (s ** 2).sum(), rtol=1e-4)
    np.testing.assert_allclose(np.abs(a.vectors).max(1), a.vectors.max(1))           # sign convention
    np.testing.assert_allclose(a.scores, (Xc @ a.vectors.T), atol=1e-3)
    torch = pytest.importorskip("torch")
    b = pca(X, k, device=torch.device("cpu"), row_chunk=100)
    np.testing.assert_allclose(b.vectors, a.vectors, atol=1e-5)
    np.testing.assert_allclose(b.scores, a.scores, atol=1e-3)
    np.testing.assert_allclose(b.eigenvalues, a.eigenvalues, rtol=1e-6)


def test_eigenvolumes_and_errors(scene):
    stack, ps, _ = scene
    sub = ParticleSet(ps.df.iloc[:10].reset_index(drop=True), APIX)
    st = build_stack(sub, stack, MASK, APIX, MSASettings(mode="plain", wedge=False, **S16))
    r = pca(st.X, 3)
    ev = eigenvolumes(st, r)
    assert ev.shape == (4, 20, 20, 20) and st.box == 20 and st.apix == 2 * APIX
    with pytest.raises(ValueError, match="needs tilt information"):
        build_stack(sub, stack, MASK, APIX, MSASettings(mode="wmd", **S16))
    with pytest.raises(ValueError, match="mode must be one of"):
        MSASettings(mode="pca")
    with pytest.raises(ValueError, match="not divisible by bin 3"):
        build_stack(sub, stack, MASK, APIX, MSASettings(mode="plain", wedge=False, bin=3))


def test_wedge_features_and_eta():
    M = euler_to_matrix(30.0, 50.0, 70.0)
    flip = np.diag([1.0, -1, -1])                    # 180° about the particle x axis: a -> -a, z -> -z
    np.testing.assert_allclose(wedge_features(M[None]), wedge_features((M @ flip)[None]))
    y = np.array([1.0, 1.1, 0.9, 5.0, 5.1, 4.9])
    assert correlation_ratio(y, ["a"] * 3 + ["b"] * 3) > 0.99
    assert correlation_ratio(y, ["a", "b"] * 3) == 0.0                              # adjusted: below chance
    assert correlation_ratio(y, ["a", "b"] * 3, adjusted=False) > 0.3
    rng = np.random.default_rng(0)
    raw, adj = np.array([(multiple_r(y, F, adjusted=False), multiple_r(y, F)) for y, F in
                         ((rng.normal(size=30), rng.normal(size=(30, 12))) for _ in range(200))]).T
    assert raw.mean() > 0.55 and np.median(adj) < 0.1                             # adjusted: ~0 at chance
