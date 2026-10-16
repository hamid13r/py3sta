import numpy as np
import pytest

from py3sta.cluster import ari, by_size, gmm, kmeans, run, stability, ward, ward_tree


def blobs(sizes=(60, 40, 25), d=4, sep=6.0, seed=0, aniso=False):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for j, n in enumerate(sizes):
        c = np.zeros(d)
        c[j % d] = sep
        c[(j + 1) % d] = sep * (j % 2)
        x = rng.normal(size=(n, d))
        if aniso:
            x = x @ np.diag([3.0] + [0.5] * (d - 1))
        X.append(c + x)
        y.append(np.full(n, j + 1))
    return np.vstack(X), np.concatenate(y)


def test_ari_and_by_size():
    assert ari([0, 0, 1, 1], [0, 0, 1, 2]) == pytest.approx(4 / 7)
    assert ari([0, 0, 1, 1, 2], [5, 5, 3, 3, 9]) == pytest.approx(1.0)
    assert ari([0] * 4, [0] * 4) == 1.0
    lab, order = by_size(np.array([2, 2, 0, 1, 1, 1]))
    assert lab.tolist() == [2, 2, 3, 1, 1, 1] and order.tolist() == [1, 2, 0]


@pytest.mark.parametrize("method", ["kmeans", "ward", "gmm"])
def test_methods_recover_blobs(method):
    X, y = blobs()
    out, Z = run(X, method, [3])
    res = out[3][0]
    assert ari(res.labels, y) > 0.95, method
    assert res.counts.tolist() == sorted(res.counts.tolist(), reverse=True)                 # 1 = largest
    assert (Z is not None) == (method == "ward")


def test_kmeans_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    X, _ = blobs(seed=2)
    a = kmeans(X, 4, n_init=5, seed=3)
    b = kmeans(X, 4, n_init=5, seed=3, device=torch.device("cpu"))
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_allclose(a.extra["centres"], b.extra["centres"], atol=1e-9)
    assert a.score == pytest.approx(b.score)


def test_gmm_bic_and_anisotropic():
    X, y = blobs(sizes=(150, 100, 80), aniso=True, sep=5.0, seed=4)
    bic = {k: gmm(X, k).bic for k in range(1, 6)}
    assert min(bic, key=bic.get) == 3, bic
    g = gmm(X, 3)
    assert ari(g.labels, y) > 0.9
    assert ((g.prob > 0) & (g.prob <= 1)).all()


def test_ward_tree_cuts():
    X, y = blobs()
    Z = ward_tree(X)
    for k in (2, 3, 4):
        assert ward(X, k, Z).k == k
    assert ward(X, 3, Z).score > ward(X, 4, Z).score            # higher merges for fewer classes


def test_stability_flags_noise_classes():
    X, y = blobs(seed=5)
    out, _ = run(X, "kmeans", [3, 6], n_boot=10)
    s3, s6 = out[3][1], out[6][1]
    assert s3.ari_mean > 0.95 and s3.jaccard.min() > 0.9
    assert s6.ari_mean < s3.ari_mean and s6.jaccard.min() < 0.8
    st = stability(X, y, lambda Z: kmeans(Z, 3).labels, n_boot=3)
    assert len(st.ari) == 3 and st.jaccard.shape == (3,)


def test_hdbscan_finds_k_and_noise():
    pytest.importorskip("sklearn")
    from py3sta.cluster import hdbscan

    X, y = blobs(sizes=(80, 60, 40), seed=6)
    rng = np.random.default_rng(0)
    X = np.vstack([X, rng.uniform(-10, 16, (10, X.shape[1]))])
    r = hdbscan(X, min_cluster_size=15)
    assert r.k == 3 and ari(r.labels[:180], y) > 0.9
    assert r.extra["n_noise"] >= 5 and (r.labels[r.labels > 0] >= 1).all()
    assert r.counts.tolist() == sorted(r.counts.tolist(), reverse=True)


def test_umap_embed_and_cramers_v():
    pytest.importorskip("umap")
    from py3sta.cluster import cramers_v, umap_embed

    X, y = blobs(seed=7)
    E = umap_embed(X, n_neighbors=15, n_components=2, seed=1)
    assert E.shape == (len(X), 2)
    np.testing.assert_allclose(umap_embed(X, n_neighbors=15, n_components=2, seed=1), E)       # reproducible
    assert ari(kmeans(E, 3).labels, y) > 0.95
    assert cramers_v(y, y) == pytest.approx(1.0) and cramers_v(y, np.zeros_like(y)) == 0.0
    assert cramers_v(y, np.random.default_rng(0).integers(0, 3, len(y))) < 0.25
