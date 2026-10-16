"""Diagnostic PNGs written by the pipeline steps (matplotlib, optional: steps skip plots without it).
Interactive views belong to the dashboard (PLAN.md §4); these are quick looks for the terminal user."""
from __future__ import annotations

import numpy as np


def available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def scree(path, var_fraction, title: str = "") -> None:
    plt = _plt()
    v = np.asarray(var_fraction) * 100
    k = np.arange(1, len(v) + 1)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(k, v, color="tab:blue", label="per factor")
    ax2 = ax.twinx()
    ax2.plot(k, np.cumsum(v), "o-", color="tab:orange", ms=3, label="cumulative")
    ax.set_xlabel("factor")
    ax.set_ylabel("variance (%)")
    ax2.set_ylabel("cumulative (%)")
    ax.set_xticks(k)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def volume_views(path, vols, labels, signed=None, title: str = "", project: bool = True) -> None:
    """xy / xz / yz views (rows) of each volume (columns): projections along z / y / x (``project``; an
    off-centre feature is still seen) or central slices. ``signed``: per volume, a symmetric red-blue
    scale (eigenvolumes) instead of grey."""
    plt = _plt()
    vols = [np.asarray(v) for v in vols]
    signed = signed or [False] * len(vols)
    n = len(vols)
    fig, axes = plt.subplots(3, n, figsize=(1.3 * n + 0.4, 4.2), squeeze=False)
    for j, (v, lab, sg) in enumerate(zip(vols, labels, signed)):
        c = v.shape[0] // 2
        if project:                                       # [z, y, x]: xy, xz, yz
            sl = (v.sum(0), v.sum(1), v.sum(2))
        else:
            sl = (v[c], v[:, c], v[:, :, c])
        lim = float(max(np.abs(x).max() for x in sl)) or 1.0
        for i, s in enumerate(sl):
            ax = axes[i, j]
            if sg:
                ax.imshow(s, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
            else:
                ax.imshow(s, cmap="gray", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(lab, fontsize=8)
            if j == 0:
                ax.set_ylabel(("xy", "xz", "yz")[i], fontsize=8)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def factor_bars(path, table, columns, threshold: float | None = None, title: str = "") -> None:
    """Grouped bars of diagnostic columns (e.g. wedge_r, tomo_eta) per factor."""
    plt = _plt()
    cols = [c for c in columns if c in table]
    k = table["factor"].to_numpy()
    w = 0.8 / max(len(cols), 1)
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(k) * max(len(cols), 1)), 3.5))
    for i, c in enumerate(cols):
        ax.bar(k + (i - (len(cols) - 1) / 2) * w, table[c].to_numpy(), w, label=c)
    if threshold is not None:
        ax.axhline(threshold, color="k", ls="--", lw=0.8)
    ax.set_xticks(k)
    ax.set_ylim(0, 1)
    ax.set_xlabel("factor")
    ax.set_ylabel("correlation")
    ax.legend(fontsize=8, ncol=len(cols))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
