import json

import numpy as np
import pandas as pd
import pytest

from py3sta.cluster import ari
from py3sta.io.mrc import read_mrc
from py3sta.io.star import read_star
from py3sta.pipeline import ConfigError, load_project
from py3sta.pipeline.config import ClassifyConfig, resolve_use
from py3sta.pipeline.manifest import read_manifest
from py3sta.pipeline.runner import run_cycle, step_state

from test_pipeline_msa import two_state_proj  # noqa: F401  (fixture)
from test_pipeline_run import _edit_yaml

RUNS = {"km": {"k": [2, 3]},
        "umap_km": {"space": "umap", "k": [2], "umap": {"n_neighbors": 8, "n_components": 2}},
        "hdb": {"method": "hdbscan", "space": "umap", "min_cluster_size": 5,
                "umap": {"n_neighbors": 8, "n_components": 2}, "averages": False},
        "truth": {"method": "column", "column": "rlnRandomSubset"}}


def _truth(proj, names):
    t = read_star(proj.cycle_input(0))["particles"].set_index("rlnTomoParticleName")["rlnRandomSubset"]
    return t.loc[names].to_numpy()


def _labels(folder, run, tag):
    df = read_star(folder / run / f"class_{tag}.star")["classes"]
    return df, df["py3staClass"].to_numpy()


def _runs(proj, runs=RUNS, **extra):
    pytest.importorskip("umap")
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["classify"].update(runs=runs, **extra))
    return load_project(proj.root)


def test_runs_compare_and_select(two_state_proj):
    proj = _runs(two_state_proj)
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["select"].update(align=[]))
    run_cycle(proj, 0, stop="classify")
    out = proj.cycle_dir(0) / "05_classify"
    for f in ("compare.star", "agreement.csv", "compare.png", "agreement.png", "km/class_K002.star",
              "km/classes_K003.mrcs", "km/summary.star", "km/scatter.png", "km/classes_K002.png", "km/run.json",
              "umap_km/embedding.npy", "umap_km/class_K002.star", "hdb/class_all.star", "truth/classes_all.mrcs"):
        assert (out / f).is_file(), f
    assert not (out / "hdb" / "classes_all.mrcs").exists()                       # averages: false
    for run, tag in (("km", "K002"), ("umap_km", "K002"), ("hdb", "all")):
        df, lab = _labels(out, run, tag)
        assert ari(lab, _truth(proj, df["rlnTomoParticleName"])) == 1.0, run
    comp = read_star(out / "compare.star")["compare"]
    assert (comp["py3staRun"] + "/" + comp["py3staTag"]).tolist() == ["km/K002", "km/K003", "umap_km/K002",
                                                                       "hdb/all", "truth/all"]
    assert comp.set_index("py3staTag").loc["K002", "py3staWedgeR"].max() < 0.3
    agree = pd.read_csv(out / "agreement.csv", index_col=0)
    assert agree.loc["km/K002", "truth/all"] == 1.0 and agree.loc["hdb/all", "umap_km/K002"] == 1.0
    info = read_manifest(out)["info"]
    assert info["cached"] == [] and info["use"] is None and info["classifications"]["hdb/all"]["k"] == 2
    # several classifications and no choice: select stops and lists them
    with pytest.raises(ValueError, match="set classify.use .* km/K002, km/K003, umap_km/K002, hdb/all, truth/all"):
        run_cycle(load_project(proj.root), 0, start="select")
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["classify"].update(use="km/K002"))
    p = load_project(proj.root)
    assert step_state(p, 0, "classify").state == "done"           # use is select's choice
    run_cycle(p, 0, start="select")
    fin = read_star(proj.cycle_dir(0) / "08_final" / "cycle-000_final.star")["particles"]
    assert ari(fin["rlnClassNumber"], fin["rlnRandomSubset"]) == 1.0
    assert read_manifest(proj.cycle_dir(0) / "07_select")["info"]["classification"] == "km/class_K002.star"
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["classify"].update(use="hdb"))
    with pytest.raises(ValueError, match="no class averages"):
        run_cycle(load_project(proj.root), 0, start="select")


def test_only_changed_runs_are_recomputed(two_state_proj):
    runs = {k: RUNS[k] for k in ("km", "truth")}
    proj = _runs(two_state_proj, runs, bootstrap=0)
    run_cycle(proj, 0, stop="classify")
    out = proj.cycle_dir(0) / "05_classify"
    stamp = {r: (out / r / "run.json").stat().st_mtime_ns for r in runs}
    # add a run and change another: only those two are computed
    runs2 = {"km": {"k": [2]}, "truth": RUNS["truth"], "gmm": {"method": "gmm", "k": [2]}}
    proj = _runs(proj, runs2)
    assert step_state(proj, 0, "classify").state == "stale"
    run_cycle(proj, 0, start="classify", stop="classify")
    info = read_manifest(out)["info"]
    assert info["cached"] == ["truth"] and info["runs"] == ["km", "truth", "gmm"]
    assert (out / "truth" / "run.json").stat().st_mtime_ns == stamp["truth"]
    assert (out / "km" / "run.json").stat().st_mtime_ns != stamp["km"]
    assert json.loads((out / "km" / "run.json").read_text())["tags"] == ["K002"]
    # removed runs leave their folder but drop out of the comparison
    proj = _runs(proj, {"gmm": {"method": "gmm", "k": [2]}})
    run_cycle(proj, 0, start="classify", stop="classify")
    assert (out / "km").is_dir() and read_star(out / "compare.star")["compare"]["py3staRun"].tolist() == ["gmm"]
    # a new msa result invalidates every run
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["msa"].update(n_factors=5))
    run_cycle(load_project(proj.root), 0, start="msa", stop="classify")
    assert read_manifest(out)["info"]["cached"] == []


def test_flat_config_is_one_run(two_state_proj):
    _edit_yaml(two_state_proj.cycle_dir(0) / "cycle.yaml", lambda c: c["classify"].update(method="ward", bootstrap=0))
    run_cycle(two_state_proj, 0, stop="classify")
    out = two_state_proj.cycle_dir(0) / "05_classify" / "ward"
    df, lab = _labels(out.parent, "ward", "K002")
    assert ari(lab, _truth(two_state_proj, df["rlnTomoParticleName"])) == 1.0
    assert (out / "linkage.npy").is_file() and (out / "dendrogram.png").is_file()
    assert read_mrc(out / "classes_K003.mrcs")[0].shape == (3 * 40, 40, 40)


def test_resolve_use_and_validation(two_state_proj):
    runs = ClassifyConfig(runs={"a": {"k": [2, 4]}, "b": {"method": "hdbscan"}}).resolved_runs()
    assert resolve_use("a/K004", runs) == ("a", "K004") == resolve_use("a/4", runs)
    assert resolve_use("b", runs) == ("b", "all") == resolve_use("b/all", runs)
    for use, msg in (("a", "several classifications"), ("a/3", "no K=3"), ("c/K002", "no run 'c'"), (2, "several runs"),
                     (None, "set classify.use")):
        with pytest.raises(ValueError, match=msg):
            resolve_use(use, runs)
    one = ClassifyConfig(k=[2, 3]).resolved_runs()
    assert resolve_use(3, one) == ("kmeans", "K003") == resolve_use("K003", one)
    auto = ClassifyConfig(runs=[{"factors": [1, 2, 3]}, {"space": "umap", "factors": [1, 3]},
                                {"method": "column", "column": "x"}]).resolved_runs()
    assert list(auto) == ["kmeans_f1-3", "kmeans_umap_f1_3", "column_x"]
    cy = two_state_proj.cycle_dir(0) / "cycle.yaml"
    for upd, msg in (({"runs": {"a": {"factors": [1, 7]}}}, "classify.runs.a.factors"),
                     ({"runs": {"a": {"k": [1, 2]}}}, "classify.runs.a.k"),
                     ({"runs": {"a": {"kk": 2}}}, "classify.runs.a.kk: unknown key"),
                     ({"runs": {"a": {"method": "hdbscan", "min_cluster_size": 1}}}, "min_cluster_size"),
                     ({"runs": {"a": {"space": "tsne"}}}, "classify.runs.a.space"),
                     ({"runs": {"a b": {}}}, "letters, digits"),
                     ({"runs": {"a": {}}, "use": "a/K005"}, "classify.use"),
                     ({"method": "dbscan"}, "classify.method")):
        _edit_yaml(cy, lambda c: c["classify"].update(**upd))
        with pytest.raises(ConfigError, match=msg):
            load_project(two_state_proj.root).cycle_config(0)
        _edit_yaml(cy, lambda c: c.update(classify={"factors": [1, 2, 3], "k": [2, 3]}))


def test_warns_on_wedge_factors(two_state_proj):
    proj = two_state_proj
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml",
               lambda c: (c["msa"].update(mode="plain", wedge=False), c["classify"].update(bootstrap=0, k=[2])))
    run_cycle(proj, 0, stop="classify")
    w = read_manifest(proj.cycle_dir(0) / "05_classify")["warnings"]
    assert any("flagged by msa as following the missing wedge" in x for x in w), w
