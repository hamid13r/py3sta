import numpy as np
import pytest

from py3sta.cluster import ari
from py3sta.io.mrc import read_mrc
from py3sta.io.star import read_star
from py3sta.pipeline import ConfigError, load_project
from py3sta.pipeline.manifest import read_manifest
from py3sta.pipeline.runner import run_cycle, step_state

from test_pipeline_msa import two_state_proj  # noqa: F401  (fixture)
from test_pipeline_run import _edit_yaml


def _truth(proj, names):
    t = read_star(proj.cycle_input(0))["particles"].set_index("rlnTomoParticleName")["rlnRandomSubset"]
    return t.loc[names].to_numpy()


def test_kmeans_classes_to_select_and_final(two_state_proj):
    proj = two_state_proj
    cy = proj.cycle_dir(0) / "cycle.yaml"
    _edit_yaml(cy, lambda c: c["select"].update(align=[]))
    run_cycle(proj, 0, stop="classify")
    out = proj.cycle_dir(0) / "05_classify"
    for f in ("class_kmeans_K002.star", "class_kmeans_K003.star", "classes_K002.mrcs", "classes_K002.star",
              "classes_K003.star", "summary.star", "scatter.png", "classes_K002.png", "summary.png"):
        assert (out / f).is_file(), f
    df = read_star(out / "class_kmeans_K002.star")["classes"]
    assert ari(df["py3staClass"], _truth(proj, df["rlnTomoParticleName"])) == 1.0
    assert "py3staClassDist" in df
    classes = read_star(out / "classes_K002.star")["classes"]
    assert classes["py3staClassCount"].sum() == 32 and (classes["py3staClassJaccard"] > 0.9).all()
    assert read_mrc(out / "classes_K003.mrcs")[0].shape == (3 * 40, 40, 40)
    info = read_manifest(out)["info"]
    assert info["per_k"]["2"]["ari"] == 1.0 and 2 in info["stable_k"]
    # several K and no choice: select stops and asks for classify.use
    with pytest.raises(ValueError, match="set classify.use"):
        run_cycle(load_project(proj.root), 0, start="select")
    _edit_yaml(cy, lambda c: c["classify"].update(use=2))
    p = load_project(proj.root)
    assert step_state(p, 0, "classify").state == "done"           # use is select's choice
    run_cycle(p, 0, start="select")
    fin = read_star(proj.cycle_dir(0) / "08_final" / "cycle-000_final.star")["particles"]
    assert ari(fin["rlnClassNumber"], fin["rlnRandomSubset"]) == 1.0
    assert read_manifest(proj.cycle_dir(0) / "07_select")["info"]["classification"] == "class_kmeans_K002.star"
    _edit_yaml(cy, lambda c: c["classify"].update(use=3))
    st = step_state(load_project(proj.root), 0, "select")
    assert st.state == "stale" and any("classification" in r for r in st.reasons), st.reasons


@pytest.mark.parametrize("method", ["ward", "gmm"])
def test_ward_and_gmm(two_state_proj, method):
    proj = two_state_proj
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["classify"].update(method=method, bootstrap=0))
    run_cycle(proj, 0, stop="classify")
    out = proj.cycle_dir(0) / "05_classify"
    df = read_star(out / f"class_{method}_K002.star")["classes"]
    assert ari(df["py3staClass"], _truth(proj, df["rlnTomoParticleName"])) == 1.0
    info = read_manifest(out)["info"]
    assert "stable_k" not in info
    if method == "ward":
        assert (out / "linkage.npy").is_file() and (out / "dendrogram.png").is_file()
    else:
        assert "py3staClassProb" in df and info["bic_k"] in (2, 3)


def test_classify_validation(two_state_proj):
    cy = two_state_proj.cycle_dir(0) / "cycle.yaml"
    for upd, msg in (({"factors": [1, 7]}, "classify.factors"), ({"k": [1, 2]}, "classify.k"),
                     ({"use": 4}, "classify.use"), ({"method": "dbscan"}, "classify.method")):
        _edit_yaml(cy, lambda c: c["classify"].update(**upd))
        with pytest.raises(ConfigError, match=msg):
            load_project(two_state_proj.root).cycle_config(0)
        _edit_yaml(cy, lambda c: c.update(classify={"factors": [1, 2, 3], "k": [2, 3]}))
