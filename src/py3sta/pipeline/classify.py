"""05_classify: named classification runs (config.ClassifyRun), compared side by side.

Each run (a method on chosen MSA factors, in PCA or UMAP space, for a list of K; or an integer STAR column) has
its own folder ``05_classify/<run>/``:

  class_<tag>.star       per particle: py3staClass (+ py3staClassDist / py3staClassProb); tag K004 or 'all'
  classes_<tag>.star     per class: count, fraction, FSC 0.143 resolution of its average, bootstrap Jaccard
  classes_<tag>.mrcs     class averages (from the stack, with the cycle's averaging settings; averages: true)
  wedge.star             wedge: whiten — fraction of each factor's variance removed (py3staWedgeRemoved)
  summary.star           per classification: stability (ARI, weakest Jaccard), class/wedge r, class/tomogram V,
                         best class resolution, BIC (gmm), noise (hdbscan)
  embedding.npy          UMAP coordinates (space: umap); linkage.npy (ward); PNGs
  run.json               settings + upstream key: the run is recomputed only when this key changes

The step folder holds the comparison of all runs in the config: compare.star (one row per classification),
agreement.csv (ARI between every pair of classifications), compare.png, agreement.png. ``classify.use`` names
the classification passed to select (config.resolve_use).

The run folders are managed here, not as step outputs (run_step removes registered outputs before a re-run;
that would defeat the per-run cache). Delete a run folder to force its recomputation.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil

import numpy as np

from .config import ClassifyRun, CycleConfig, ProjectConfig, resolve_use, run_tags

CLASS_COLUMN = "py3staClass"
STABLE_JACCARD = 0.6                                       # a class below this is not reproducible
WEDGE_CLASS_R = 0.3                                        # warn: classes follow the wedge orientation
RUN_FILE = "run.json"
NO_MSA = ("column", "groups")                              # methods that do not read MSA factors


# ------------------------------------------------------------------ params
def run_params(r: ClassifyRun) -> dict:
    """The settings that change a run's result (irrelevant ones dropped, so editing them changes nothing)."""
    d = dataclasses.asdict(r)
    if r.method == "column":
        return {"method": "column", "column": r.column, "averages": r.averages}
    if r.method == "groups":
        return {"method": "groups", "file": r.file, "averages": r.averages}
    d.pop("column")
    d.pop("file")
    if r.method == "hdbscan":
        d.pop("k")
        d.pop("n_init")
    else:
        d.pop("min_cluster_size")
        d.pop("min_samples")
    if r.space != "umap":
        d.pop("umap")
    return d


def classify_upstream(c: CycleConfig) -> tuple:
    runs = c.classify.resolved_runs()
    return () if all(r.method in NO_MSA for r in runs.values()) else ("msa",)


def classify_params(p: ProjectConfig, c: CycleConfig) -> dict:
    from .steps import _source_name, average_params

    runs = c.classify.resolved_runs()                      # 'use' is select's choice: not a classify parameter
    return {"classify": {name: run_params(r) for name, r in runs.items()}, **average_params(p, c),
            "pose_source": _source_name("classify")}


def classification_files(c: CycleConfig) -> tuple[str, str, str]:
    """(per-particle class STAR, classes STAR, class averages) of the classification select uses, relative
    to 05_classify/."""
    run, tag = resolve_use(c.classify.use, c.classify.resolved_runs())
    return f"{run}/class_{tag}.star", f"{run}/classes_{tag}.star", f"{run}/classes_{tag}.mrcs"


# ----------------------------------------------------------------- helpers
def groups_file(project, n: int, name: str):
    """A groups JSON: absolute, or relative to the cycle folder, the project, or the working directory."""
    from pathlib import Path

    for c in (Path(name), project.cycle_dir(n) / name, project.root / name):
        if c.is_file():
            return c
    raise FileNotFoundError(f"{name}: groups file not found (relative to {project.cycle_dir(n)}, {project.root} or "
                            f"{Path.cwd()}); export it from 'py3sta explore'")


def _class_averages(project, cin, cls, ids, average, path):
    """Class averages written to ``path`` (or None: counts only) and per-class rows."""
    from ..average import fsc, resolution_at
    from ..io.mrc import open_volume_stack

    p = project.config
    df = cin.particles.df
    out = open_volume_stack(path, len(ids), p.box, np.float32, p.apix, mode="w+") if path is not None else None
    rows, vols = [], []
    for k, cid in enumerate(ids):
        sel = cls == cid
        row = {CLASS_COLUMN: int(cid), "py3staClassIndex": k + 1, "py3staClassCount": int(sel.sum()),
               "py3staClassFraction": round(float(sel.mean()), 4), "py3staClassRes": np.nan}
        if out is not None:
            mdf = df[sel].reset_index(drop=True)
            if "rlnRandomSubset" in mdf and mdf["rlnRandomSubset"].nunique() < 2:
                mdf = mdf.drop(columns="rlnRandomSubset")     # one half set only: FSC halves = alternate rows
            members = dataclasses.replace(cin.particles, df=mdf)
            res = average(members)
            out[k] = res.average
            vols.append(res.average)
            if res.halves is not None and sel.sum() >= 4:
                r, f = fsc(*res.halves, res.apix)
                row["py3staClassRes"] = round(float(resolution_at(r, f)), 1)
        rows.append(row)
    if out is not None:
        out.flush()
    return rows, vols


def _column_labels(ctx, cin, column: str, path):
    import pandas as pd

    df = cin.particles.df
    if column not in df:
        ctx.warn(f"no column {column}: all particles in class 1")
        return np.ones(len(df), int)
    v = pd.to_numeric(df[column], errors="coerce").to_numpy()
    bad = ~np.isfinite(v) | (v != np.round(v))
    if bad.any():
        raise ValueError(f"{path.name}: column {column} is not an integer for {int(bad.sum())} particle(s)")
    return v.astype(int)


def _key(r: ClassifyRun, upstream: dict) -> str:
    text = json.dumps({"run": run_params(r), "upstream": upstream}, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def _cached(rdir, key: str):
    f = rdir / RUN_FILE
    if not f.is_file():
        return None
    try:
        rec = json.loads(f.read_text())
    except ValueError:
        return None
    if rec.get("key") != key:
        return None
    needed = [f"class_{t}.star" for t in rec["tags"]] + [f"classes_{t}.star" for t in rec["tags"]]
    if rec.get("averages"):
        needed += [f"classes_{t}.mrcs" for t in rec["tags"]]
    return rec if all((rdir / n).is_file() for n in needed) else None


# --------------------------------------------------------------------- run
def _compute_run(ctx, project, n: int, name: str, r: ClassifyRun, rdir, cin, path, scores, average, device):
    """Classify, write the run folder, return the summary rows."""
    import pandas as pd

    from .. import cluster, plots
    from ..io.star import write_star
    from ..msa import MIN_PER_WEDGE_FEATURE, class_wedge_r, whiten_wedge
    from .project import PARTICLE_NAME

    names = cin.particles.df[PARTICLE_NAME].astype(str).to_numpy()
    tomos = cin.particles.df["rlnTomoName"].to_numpy()
    mats = cin.particles.matrices
    Z = None
    view, view_axes = None, None
    class_names = {}
    wedge_removed = np.nan                                # fraction of the factors' variance removed by whitening
    if r.method == "column":
        lab = _column_labels(ctx, cin, r.column, path)
        results = {"all": (cluster.ClusterResult(lab, int(len(np.unique(lab))), "column", 0.0), None)}
        ctx.log(f"run {name}: classes from column {r.column}")
    elif r.method == "groups":
        from .explore import load_groups

        gfile = groups_file(project, n, r.file)
        lab, gnames = load_groups(gfile, names)
        class_names = {i + 1: g for i, g in enumerate(gnames)}
        class_names[0] = "(not in a group)"
        results = {"all": (cluster.ClusterResult(lab, len(gnames), "groups", 0.0), None)}
        ctx.log(f"run {name}: {len(gnames)} group(s) from {gfile.name}: "
                + ", ".join(f"{g} ({int((lab == i + 1).sum())})" for i, g in enumerate(gnames))
                + f"; not in a group: {int((lab == 0).sum())}")
    else:
        X = scores[[f"py3staPC{f}" for f in r.factors]].to_numpy(float)
        if r.wedge == "whiten":
            if len(X) < MIN_PER_WEDGE_FEATURE * 12:
                ctx.warn(f"run {name}: wedge whitening with only {len(X)} particles (< {MIN_PER_WEDGE_FEATURE * 12}): "
                         "the fit is noisy and may remove real signal (wedge: keep to compare)")
            var = X.var(0)
            X, removed = whiten_wedge(X, mats)
            wedge_removed = float((removed * var).sum() / max(var.sum(), 1e-30))
            write_star({"wedge": pd.DataFrame({"py3staFactor": list(r.factors),
                                               "py3staWedgeRemoved": np.round(removed, 4)})}, rdir / "wedge.star")
            ctx.log(f"run {name}: wedge whitening removed {100 * wedge_removed:.1f} % of the factors' variance "
                    "(per factor: " + ", ".join(f"{f} {100 * v:.0f} %" for f, v in zip(r.factors, removed)) + ")")
        if r.standardize:
            X = X / np.where(X.std(0) > 0, X.std(0), 1)
        Y = X
        view, view_axes = X[:, :2] if X.shape[1] > 1 else np.column_stack([X[:, 0], X[:, 0]]), \
            tuple(f"PC{f}" for f in (r.factors[:2] if len(r.factors) > 1 else r.factors * 2))
        if r.space == "umap":
            u = r.umap
            Y = cluster.umap_embed(X, u.n_neighbors, u.min_dist, u.n_components, u.metric, r.seed)
            np.save(rdir / "embedding.npy", Y)
            if u.n_components >= 2:
                view, view_axes = Y[:, :2], ("UMAP1", "UMAP2")
        ctx.log(f"run {name}: {r.method} on factors {r.factors}" + (" (UMAP space)" if r.space == "umap" else "")
                + (f", K = {r.k}" if r.method != "hdbscan" else f", min_cluster_size {r.min_cluster_size}")
                + (f", stability from {r.bootstrap} subsets" if r.bootstrap else ""))
        if r.method == "hdbscan":
            res = cluster.hdbscan(Y, r.min_cluster_size, r.min_samples)

            def fit(S):
                return cluster.hdbscan(S, r.min_cluster_size, r.min_samples).labels
            st = cluster.stability(Y, res.labels, fit, r.bootstrap, seed=r.seed) if r.bootstrap and res.k else None
            results = {"all": (res, st)}
        else:
            out, Z = cluster.run(Y, r.method, r.k, n_init=r.n_init, seed=r.seed,
                                 device=device if r.method == "kmeans" else None, n_boot=r.bootstrap)
            results = {f"K{k:03d}": v for k, v in out.items()}
        if Z is not None:
            np.save(rdir / "linkage.npy", Z)

    summary, montage = [], {}
    for tag, (res, st) in results.items():
        table = pd.DataFrame({PARTICLE_NAME: names, CLASS_COLUMN: res.labels})
        if res.dist is not None:
            table["py3staClassDist"] = np.round(res.dist, 5)
        if res.prob is not None:
            table["py3staClassProb"] = np.round(res.prob, 5)
        write_star({"classes": table}, rdir / f"class_{tag}.star")
        ids = np.unique(res.labels)
        rows, vols = _class_averages(project, cin, res.labels, ids, average,
                                     rdir / f"classes_{tag}.mrcs" if r.averages else None)
        if class_names:
            for row in rows:
                row["py3staClassName"] = class_names.get(row[CLASS_COLUMN], "")
        if st is not None:
            for row in rows:
                c = row[CLASS_COLUMN]
                row["py3staClassJaccard"] = round(float(st.jaccard[c - 1]), 3) if c >= 1 else np.nan
        write_star({"classes": pd.DataFrame(rows)}, rdir / f"classes_{tag}.star")
        if vols:
            montage[tag] = (vols, [f"{row[CLASS_COLUMN]} ({row['py3staClassCount']})" for row in rows])
        n_classes = int((ids > 0).sum())
        wedge_r = class_wedge_r(res.labels, mats)
        res_vals = [row["py3staClassRes"] for row in rows if row[CLASS_COLUMN] > 0 and np.isfinite(row["py3staClassRes"])]
        srow = {"py3staTag": tag, "py3staK": n_classes, "py3staNoise": int((res.labels == 0).sum()),
                "py3staSmallestClass": int(min(row["py3staClassCount"] for row in rows if row[CLASS_COLUMN] > 0))
                if n_classes else 0,
                "py3staScore": round(float(res.score), 4), "py3staWedgeR": round(wedge_r, 3),
                "py3staWedgeRemoved": round(wedge_removed, 4) if np.isfinite(wedge_removed) else np.nan,
                "py3staTomoV": round(cluster.cramers_v(res.labels, tomos), 3),
                "py3staBestRes": min(res_vals) if res_vals else np.nan,
                "py3staARI": round(st.ari_mean, 3) if st is not None else np.nan,
                "py3staJaccardMin": round(float(np.nanmin(st.jaccard)), 3) if st is not None else np.nan,
                "py3staBIC": round(res.bic, 1) if res.bic is not None else np.nan}
        summary.append(srow)
        if wedge_r > WEDGE_CLASS_R:
            ctx.warn(f"run {name} {tag}: classes follow the missing-wedge orientation (r {wedge_r:.2f})")
        ctx.log(f"  {tag}: counts {[row['py3staClassCount'] for row in rows]}"
                + (f" ({'not in a group' if r.method == 'groups' else 'noise'} = class 0: {srow['py3staNoise']})"
                   if srow["py3staNoise"] else "")
                + (f", ARI {st.ari_mean:.2f}, weakest Jaccard {np.nanmin(st.jaccard):.2f}" if st is not None else "")
                + f", wedge r {wedge_r:.2f}, tomo V {srow['py3staTomoV']:.2f}"
                + (f", best class {srow['py3staBestRes']:.1f} Å" if res_vals else "")
                + (f", BIC {res.bic:.0f}" if res.bic is not None else ""))
    sm = pd.DataFrame(summary)
    write_star({"summary": sm}, rdir / "summary.star")
    if plots.available():
        title = f"cycle {n:03d}, run {name}"
        if view is not None:
            plots.class_scatter(rdir / "scatter.png", view, {t: res.labels for t, (res, _) in results.items()},
                                view_axes, f"classes: {title}")
        for tag, (vols, labels) in montage.items():
            plots.volume_views(rdir / f"classes_{tag}.png", vols, labels, title=f"class averages (projections), {tag}: {title}")
        if r.method not in NO_MSA and (r.bootstrap or r.method == "gmm"):
            plots.stability_summary(rdir / "summary.png", sm, f"per K: {title}")
        if Z is not None:
            plots.dendrogram(rdir / "dendrogram.png", Z, r.k, f"Ward tree: {title}")
    return summary, list(results)


def run_classify(ctx, project, n: int, c: CycleConfig) -> None:
    """Every run of ``classify`` (recomputed only when its settings or upstream changed), then the comparison."""
    import pandas as pd

    from .. import cluster, plots
    from ..io.star import read_star, write_star
    from .manifest import fingerprint, read_manifest
    from .project import PARTICLE_NAME, load_cycle_input
    from .steps import _averager, _particle_block, average_params, get_step, pose_star, resolve_device

    p = project.config
    runs = c.classify.resolved_runs()
    path, src = pose_star(project, n, "classify")
    ctx.read_star(path)
    cin = load_cycle_input(project, n, path)
    names = cin.particles.df[PARTICLE_NAME].astype(str).to_numpy()
    upstream = {"poses": fingerprint(path)["sha1"], "average": average_params(p, c)}
    scores, flagged = None, set()
    if any(r.method not in NO_MSA for r in runs.values()):
        msa_star = project.cycle_dir(n) / get_step("msa").folder / "msa.star"
        if not msa_star.is_file():
            raise FileNotFoundError(f"{msa_star}: run the msa step first (msa.enable: true)")
        ctx.read_star(msa_star)
        m = read_manifest(msa_star.parent) or {}
        upstream["msa"] = m.get("run_id")
        flagged = set(m.get("info", {}).get("wedge_factors", []))
        sc = _particle_block(read_star(msa_star))
        index = pd.Series(np.arange(len(sc)), index=sc[PARTICLE_NAME].astype(str))
        if set(index.index) != set(names):
            raise ValueError(f"{msa_star.name} does not cover the active particles of {path.name} (re-run msa)")
        scores = sc.iloc[index.loc[names].to_numpy()].reset_index(drop=True)
        for name, r in runs.items():
            if r.method in NO_MSA:
                continue
            missing = [f for f in r.factors if f"py3staPC{f}" not in scores]
            if missing:
                raise ValueError(f"run {name}: msa.star has no factor(s) {missing} (msa computed fewer factors)")
            bad = sorted(set(r.factors) & flagged)
            if bad:
                ctx.warn(f"run {name}: factors {bad} were flagged by msa as following the missing wedge: "
                         + ("whitened (their linear wedge part is removed); check the classes' WedgeR"
                            if r.wedge == "whiten" else
                            "classes may sort particles by orientation (wedge: whiten, or drop them from factors)"))
    device, label = resolve_device(p.compute)
    average = None
    folder = ctx.folder
    done, cached = {}, []
    for name, r in runs.items():
        rdir = folder / name
        up = upstream
        if r.method == "groups":
            gfile = groups_file(project, n, r.file)
            ctx.use_files([gfile], "groups")
            up = {**upstream, "groups": hashlib.sha1(gfile.read_bytes()).hexdigest()}
        key = _key(r, up)
        rec = _cached(rdir, key)
        if rec is not None:
            ctx.log(f"run {name}: unchanged (cached)")
            cached.append(name)
            done[name] = rec
            continue
        if rdir.exists():
            shutil.rmtree(rdir)
        rdir.mkdir(parents=True)
        if average is None and any(x.averages for x in runs.values()):
            average = _averager(ctx, project, c, cin)
        summary, tags = _compute_run(ctx, project, n, name, r, rdir, cin, path, scores, average, device)
        rec = {"key": key, "run": run_params(r), "tags": tags, "averages": r.averages, "summary": summary}
        (rdir / RUN_FILE).write_text(json.dumps(rec, indent=1, default=float))
        done[name] = rec

    # comparison of every run in the config
    rows, labels = [], {}
    for name, rec in done.items():
        r = runs[name]
        for s in rec["summary"]:
            rows.append({"py3staRun": name, "py3staMethod": r.method,
                         "py3staSpace": r.method if r.method in NO_MSA else r.space,
                         "py3staFactors": "" if r.method in NO_MSA else ",".join(map(str, r.factors)),
                         "py3staWedge": "" if r.method in NO_MSA else r.wedge, **s})
            cs = _particle_block(read_star(folder / name / f"class_{s['py3staTag']}.star"))
            lab = cs.set_index(cs[PARTICLE_NAME].astype(str))[CLASS_COLUMN]
            labels[f"{name}/{s['py3staTag']}"] = lab.loc[names].to_numpy()
    comp = pd.DataFrame(rows)
    write_star({"compare": comp}, ctx.output_star("compare.star"))
    keys = list(labels)
    agree = np.array([[cluster.ari(labels[a], labels[b]) for b in keys] for a in keys])
    pd.DataFrame(agree, index=keys, columns=keys).round(3).to_csv(ctx.output("agreement.csv"))
    if plots.available() and len(comp):
        plots.compare_runs(ctx.output("compare.png"), comp, f"cycle {n:03d}: classifications")
        if len(keys) > 1:
            plots.agreement(ctx.output("agreement.png"), keys, agree, f"cycle {n:03d}: agreement (ARI)")
    try:
        use = "/".join(resolve_use(c.classify.use, runs))
    except ValueError:
        use = None
    ctx.info.update(pose_source=src, runs=list(runs), cached=cached, use=use,
                    classifications={f"{r_['py3staRun']}/{r_['py3staTag']}": {
                        k[6:].lower(): (None if isinstance(v, float) and not np.isfinite(v) else v)
                        for k, v in r_.items() if k in ("py3staK", "py3staARI", "py3staJaccardMin", "py3staWedgeR",
                                                       "py3staWedgeRemoved", "py3staTomoV", "py3staBestRes", "py3staBIC", "py3staNoise")}
                        for r_ in rows})
    ctx.log(f"{len(keys)} classification(s) from {len(runs)} run(s) ({len(cached)} cached); compare.star"
            + (f"; select uses {use}" if use else "; set classify.use to choose the one passed to select"))
