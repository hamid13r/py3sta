"""Classification of MSA factor scores (PLAN.md §2 step 5): k-means, Ward, GMM, stability.

Every method returns classes 1..K numbered by size (1 = largest), so numbering is stable between runs.

* ``kmeans`` -- Lloyd iterations from k-means++ seeds (drawn in numpy, so numpy and torch run the same
  restarts); best of ``n_init`` by inertia. torch on the GPU, numpy as the reference.
* ``ward`` -- Ward hierarchical clustering (scipy linkage, i3 parity): one tree, cut at every K; memory
  grows as n^2, refused above WARD_MAX particles.
* ``gmm`` -- Gaussian mixture, full covariances, EM from the k-means solution; per-particle membership
  probability and BIC (lower is better) to suggest K.
* stability -- the clustering is repeated on random 80 % subsets; ARI with the full result on the shared
  particles, and per class the mean best Jaccard overlap with any subset class (Hennig 2007): a class
  below ~0.6 is not reproducible (noise-driven).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

METHODS = ("kmeans", "ward", "gmm")
WARD_MAX = 30000


@dataclass
class ClusterResult:
    labels: np.ndarray                      # (n,) classes 1..K, 1 = largest
    k: int
    method: str
    score: float                            # kmeans: inertia; ward: merge height of the cut; gmm: log likelihood
    bic: float | None = None                # gmm
    prob: np.ndarray | None = None          # gmm: (n,) probability of the assigned class
    dist: np.ndarray | None = None          # kmeans: (n,) distance to the class centre
    extra: dict = field(default_factory=dict)

    @property
    def counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=self.k + 1)[1:]


def by_size(labels) -> tuple[np.ndarray, np.ndarray]:
    """Renumber 0-based ``labels`` to 1..K by decreasing size (ties: lower old label first);
    returns (new labels, order) with order[new - 1] = old."""
    labels = np.asarray(labels)
    ids, counts = np.unique(labels, return_counts=True)
    order = ids[np.lexsort((ids, -counts))]
    new = np.empty(labels.max() + 1, int)
    new[order] = np.arange(1, len(order) + 1)
    return new[labels], order


# ------------------------------------------------------------------ k-means
def kmeans_pp(X, k: int, rng) -> np.ndarray:
    """k-means++ seeds (indices into X)."""
    n = len(X)
    idx = [int(rng.integers(n))]
    d2 = ((X - X[idx[0]]) ** 2).sum(1)
    for _ in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        j = int(rng.choice(n, p=p))
        idx.append(j)
        d2 = np.minimum(d2, ((X - X[j]) ** 2).sum(1))
    return np.array(idx)


def _lloyd_numpy(X, C, max_iter, tol):
    for _ in range(max_iter):
        d2 = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]
        lab = d2.argmin(1)
        new = C.copy()
        for j in range(len(C)):
            m = lab == j
            if m.any():
                new[j] = X[m].mean(0)
            else:                                           # empty: the point farthest from its centre
                far = int(d2[np.arange(len(X)), lab].argmax())
                new[j] = X[far]
                lab[far] = j
        shift = ((new - C) ** 2).sum()
        C = new
        if shift <= tol:
            break
    d2 = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]
    lab = d2.argmin(1)
    return lab, C, np.maximum(d2[np.arange(len(X)), lab], 0)


def _lloyd_torch(X, C, max_iter, tol, device):
    import torch

    t = torch
    Xt = t.as_tensor(X, dtype=t.float64, device=device)
    Ct = t.as_tensor(C, dtype=t.float64, device=device)
    x2 = (Xt ** 2).sum(1)[:, None]
    k = len(C)
    for _ in range(max_iter):
        d2 = x2 - 2 * Xt @ Ct.T + (Ct ** 2).sum(1)[None]
        lab = d2.argmin(1)
        cnt = t.bincount(lab, minlength=k).to(t.float64)
        sums = t.zeros_like(Ct).index_add_(0, lab, Xt)
        new = t.where(cnt[:, None] > 0, sums / cnt.clamp_min(1)[:, None], Ct)
        if (cnt == 0).any():                                # rare: fall back to the numpy rule
            lab_np, C_np, _ = _lloyd_numpy(X, Ct.cpu().numpy(), 1, -1.0)
            new = t.as_tensor(C_np, dtype=t.float64, device=device)
        shift = float(((new - Ct) ** 2).sum())
        Ct = new
        if shift <= tol:
            break
    d2 = x2 - 2 * Xt @ Ct.T + (Ct ** 2).sum(1)[None]
    lab = d2.argmin(1)
    dist = d2.gather(1, lab[:, None])[:, 0].clamp_min(0)
    return lab.cpu().numpy(), Ct.cpu().numpy(), dist.cpu().numpy()


def kmeans(X, k: int, n_init: int = 10, max_iter: int = 300, tol: float = 1e-8, seed: int = 0,
           device=None) -> ClusterResult:
    X = np.asarray(X, np.float64)
    rng = np.random.default_rng(seed)
    best = None
    scale = float(((X - X.mean(0)) ** 2).sum()) or 1.0
    for _ in range(n_init):
        C0 = X[kmeans_pp(X, k, rng)]
        if device is None:
            lab, C, d2 = _lloyd_numpy(X, C0.copy(), max_iter, tol * scale)
        else:
            lab, C, d2 = _lloyd_torch(X, C0, max_iter, tol * scale, device)
        inertia = float(d2.sum())
        if best is None or inertia < best[0] - 1e-9 * scale:
            best = (inertia, lab, C, d2)
    inertia, lab, C, d2 = best
    labels, order = by_size(lab)
    return ClusterResult(labels, k, "kmeans", inertia, dist=np.sqrt(d2), extra={"centres": C[order]})


# --------------------------------------------------------------------- Ward
def ward_tree(X) -> np.ndarray:
    """scipy linkage matrix (Ward) of the rows of X."""
    from scipy.cluster.hierarchy import linkage

    if len(X) > WARD_MAX:
        raise ValueError(f"ward: {len(X)} particles exceed {WARD_MAX} (memory grows as n^2); use kmeans or gmm")
    return linkage(np.asarray(X, np.float64), method="ward")


def ward_cut(Z, k: int) -> ClusterResult:
    from scipy.cluster.hierarchy import fcluster

    lab = fcluster(Z, k, criterion="maxclust") - 1
    labels, _ = by_size(lab)
    height = float(Z[-(k - 1), 2]) if k > 1 else float(Z[-1, 2])  # merge undone by the cut
    return ClusterResult(labels, int(labels.max()), "ward", height)


def ward(X, k: int, Z=None) -> ClusterResult:
    return ward_cut(ward_tree(X) if Z is None else Z, k)


# ---------------------------------------------------------------------- GMM
def _log_gauss(X, mu, cov):
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (z ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * d * math.log(2 * math.pi)


def gmm(X, k: int, max_iter: int = 300, tol: float = 1e-7, reg: float = 1e-6, seed: int = 0,
        n_init: int = 10, init=None) -> ClusterResult:
    """EM for a full-covariance Gaussian mixture, started from k-means (or ``init``: 0-based labels)."""
    X = np.asarray(X, np.float64)
    n, d = X.shape
    lab = kmeans(X, k, n_init=n_init, seed=seed).labels - 1 if init is None else np.asarray(init)
    R = np.eye(k)[lab]                                     # responsibilities
    floor = reg * max(float(X.var(0).mean()), 1e-12)
    ll_old = -np.inf
    for _ in range(max_iter):
        nk = R.sum(0) + 1e-12
        w = nk / n
        mu = (R.T @ X) / nk[:, None]
        cov = np.empty((k, d, d))
        for j in range(k):
            xc = X - mu[j]
            cov[j] = (R[:, j, None] * xc).T @ xc / nk[j] + floor * np.eye(d)
        logp = np.column_stack([np.log(w[j]) + _log_gauss(X, mu[j], cov[j]) for j in range(k)])
        mx = logp.max(1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(logp - mx).sum(1))
        ll = float(lse.sum())
        R = np.exp(logp - lse[:, None])
        if ll - ll_old < tol * abs(ll):
            break
        ll_old = ll
    lab = R.argmax(1)
    labels, order = by_size(lab)
    n_par = (k - 1) + k * d + k * d * (d + 1) / 2
    bic = -2 * ll + n_par * math.log(n)
    return ClusterResult(labels, int(labels.max()), "gmm", ll, bic=bic, prob=R[np.arange(n), lab],
                         extra={"means": mu[order], "weights": w[order]})


# ---------------------------------------------------------------- stability
def ari(a, b) -> float:
    """Adjusted Rand index (Hubert & Arabie 1985)."""
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    cont = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(cont, (ai, bi), 1)

    def c2(x):
        return (x * (x - 1) / 2).sum()

    n = len(a)
    s, sa, sb = c2(cont), c2(cont.sum(1)), c2(cont.sum(0))
    exp = sa * sb / (n * (n - 1) / 2)
    mx = (sa + sb) / 2
    return 1.0 if mx == exp else float((s - exp) / (mx - exp))


@dataclass
class Stability:
    ari: np.ndarray                         # (B,) ARI with the full clustering on each subset
    jaccard: np.ndarray                     # (K,) mean best Jaccard per class (class c = index c - 1)

    @property
    def ari_mean(self) -> float:
        return float(self.ari.mean()) if len(self.ari) else float("nan")


def stability(X, labels, fit, n_boot: int = 20, frac: float = 0.8, seed: int = 0) -> Stability:
    """``fit(X_subset) -> labels`` is re-run on ``n_boot`` random subsets of ``frac`` of the particles."""
    labels = np.asarray(labels)
    K = int(labels.max())
    rng = np.random.default_rng(seed)
    n = len(labels)
    aris, jac = [], np.zeros((n_boot, K))
    for b in range(n_boot):
        idx = np.sort(rng.choice(n, int(round(frac * n)), replace=False))
        sub = np.asarray(fit(X[idx]))
        ref = labels[idx]
        aris.append(ari(ref, sub))
        for c in range(1, K + 1):
            A = ref == c
            if not A.any():
                jac[b, c - 1] = np.nan
                continue
            best = 0.0
            for s in np.unique(sub):
                B = sub == s
                best = max(best, (A & B).sum() / (A | B).sum())
            jac[b, c - 1] = best
    return Stability(np.array(aris), np.nanmean(jac, 0) if n_boot else np.full(K, np.nan))


def run(X, method: str, ks, n_init: int = 10, seed: int = 0, device=None, n_boot: int = 0):
    """{K: (ClusterResult, Stability | None)} for every K in ``ks``; Ward builds one tree for all K."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)} (got {method!r})")
    X = np.asarray(X, np.float64)
    Z = ward_tree(X) if method == "ward" else None
    out = {}
    for k in ks:
        if method == "kmeans":
            res = kmeans(X, k, n_init=n_init, seed=seed, device=device)

            def fit(Y, k=k):
                return kmeans(Y, k, n_init=max(1, n_init // 2), seed=seed + 1, device=device).labels
        elif method == "ward":
            res = ward_cut(Z, k)

            def fit(Y, k=k):
                return ward(Y, k).labels
        else:
            res = gmm(X, k, n_init=n_init, seed=seed)

            def fit(Y, k=k):
                return gmm(Y, k, n_init=max(1, n_init // 2), seed=seed + 1).labels
        st = stability(X, res.labels, fit, n_boot, seed=seed) if n_boot else None
        out[k] = (res, st)
    return out, Z
