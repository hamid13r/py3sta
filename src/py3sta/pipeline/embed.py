"""06_embed: a 2D map of the MSA factors (UMAP, t-SNE or two factors) for ``py3sta explore``.

Upstream is msa only: classification runs do not change the map (the explorer reads them when it is built).
Skipped (nothing written) when ``embed.enable`` is false or msa is disabled.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .config import CycleConfig, ProjectConfig

EMBED_FILE = "embedding.star"


def embed_upstream(c: CycleConfig) -> tuple:
    return ("msa",) if c.embed.enable and c.msa.enable else ()


def embed_params(p: ProjectConfig, c: CycleConfig) -> dict:
    return {"embed": dataclasses.asdict(c.embed), "msa_enabled": c.msa.enable}


def embed(X, method: str, n_neighbors: int = 30, min_dist: float = 0.1, perplexity: float = 30.0,
          seed: int = 0) -> np.ndarray:
    """(n, 2) coordinates: umap | tsne | pca (the first two columns of X)."""
    X = np.asarray(X, np.float64)
    if method == "pca":
        return X[:, :2].astype(np.float32)
    if method == "umap":
        from ..cluster import umap_embed

        return umap_embed(X, n_neighbors, min_dist, 2, "euclidean", seed)
    if method == "tsne":
        from sklearn.manifold import TSNE

        perp = min(perplexity, max(1.0, (len(X) - 1) / 3))
        return TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca").fit_transform(X).astype(np.float32)
    raise ValueError(f"embed method must be umap, tsne or pca (got {method!r})")


def run_embed(ctx, project, n: int, c: CycleConfig) -> None:
    import pandas as pd

    from .. import plots
    from ..io.star import read_star, write_star
    from .project import PARTICLE_NAME, load_cycle_input
    from .steps import _particle_block, get_step, pose_star

    e = c.embed
    if not e.enable or not c.msa.enable:
        ctx.info.update(enabled=False)
        ctx.log("embed skipped (" + ("embed.enable: false" if not e.enable else "msa disabled") + ")")
        return
    msa_star = project.cycle_dir(n) / get_step("msa").folder / "msa.star"
    if not msa_star.is_file():
        raise FileNotFoundError(f"{msa_star}: run the msa step first")
    ctx.read_star(msa_star)
    sc = _particle_block(read_star(msa_star))
    cols = [f"py3staPC{f}" for f in e.factors]
    missing = [col for col in cols if col not in sc]
    if missing:
        raise ValueError(f"{msa_star.name} has no {', '.join(missing)} (msa computed fewer factors)")
    X = sc[cols].to_numpy(float)
    names = sc[PARTICLE_NAME].astype(str).to_numpy()
    path, _ = pose_star(project, n, "embed")
    path = path if path.is_file() else project.cycle_input(n)
    ctx.read_star(path)                                    # poses -> wedge orientation
    df = load_cycle_input(project, n, path).particles
    pos = pd.Series(np.arange(len(df.df)), index=df.df[PARTICLE_NAME].astype(str)).loc[names].to_numpy()
    removed = None
    if e.wedge == "whiten":
        from ..msa import MIN_PER_WEDGE_FEATURE, whiten_wedge

        if len(X) < MIN_PER_WEDGE_FEATURE * 12:
            ctx.warn(f"wedge whitening with only {len(X)} particles (< {MIN_PER_WEDGE_FEATURE * 12}): the fit is noisy "
                     "and may remove real signal (embed.wedge: keep to compare)")
        var = X.var(0)
        X, per = whiten_wedge(X, df.matrices[pos])
        removed = float((per * var).sum() / max(var.sum(), 1e-30))
        ctx.log(f"wedge whitening removed {100 * removed:.1f} % of the factors' variance")
    if e.standardize:
        X = X / np.where(X.std(0) > 0, X.std(0), 1)
    ctx.log(f"{e.method} of {len(X)} particles on factors {e.factors}")
    E = embed(X, e.method, e.n_neighbors, e.min_dist, e.perplexity, e.seed)
    table = pd.DataFrame({PARTICLE_NAME: names, "py3staEmbed1": np.round(E[:, 0], 5),
                          "py3staEmbed2": np.round(E[:, 1], 5)})
    write_star({"particles": table}, ctx.output_star(EMBED_FILE))
    if plots.available():
        df = df.df.iloc[pos]
        colours = {"tomogram": ("cat", df["rlnTomoName"].to_numpy()),
                   f"PC{e.factors[0]}": ("num", X[:, 0])}
        if "py3staRefCC" in df:
            colours["alignment CC"] = ("num", df["py3staRefCC"].to_numpy(float))
        axes = ("PC%d" % e.factors[0], "PC%d" % e.factors[1]) if e.method == "pca" else (f"{e.method}1", f"{e.method}2")
        plots.embedding_panels(ctx.output("embedding.png"), E, colours, axes, f"cycle {n:03d}: {e.method} of factors {e.factors}")
    ctx.info.update(enabled=True, method=e.method, factors=list(e.factors), n_particles=len(X), wedge=e.wedge,
                    wedge_removed=removed)
