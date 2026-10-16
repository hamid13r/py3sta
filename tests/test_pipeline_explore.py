import json
import re
import shutil
import subprocess

import numpy as np
import pytest
from click.testing import CliRunner

from py3sta.cli import main
from py3sta.cluster import ari
from py3sta.io.star import read_star
from py3sta.pipeline import ConfigError, load_project
from py3sta.pipeline.explore import build_explore, load_groups, write_groups
from py3sta.pipeline.manifest import read_manifest
from py3sta.pipeline.runner import run_cycle, step_state

from test_pipeline_msa import two_state_proj  # noqa: F401  (fixture)
from test_pipeline_run import _edit_yaml

EMBED = {"n_neighbors": 8}
RUNS = {"km": {"k": [2], "bootstrap": 0},
        "umap_km": {"space": "umap", "k": [2], "bootstrap": 0, "umap": {"n_neighbors": 8, "n_components": 2}}}


def _setup(proj, **embed):
    pytest.importorskip("umap")
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml",
               lambda c: (c["embed"].update(**{**EMBED, **embed}), c["classify"].update(runs=RUNS)))
    return load_project(proj.root)


def _embedding(proj):
    df = read_star(proj.cycle_dir(0) / "06_embed" / "embedding.star")["particles"]
    truth = read_star(proj.cycle_input(0))["particles"].set_index("rlnTomoParticleName")["rlnRandomSubset"]
    return df, truth.loc[df["rlnTomoParticleName"]].to_numpy()


def _inside(points, poly):
    """Ray casting point-in-polygon (what the browser lasso does)."""
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        cross = (y1 > y) != (y2 > y)
        xi = x1 + (y - y1) * (x2 - x1) / np.where(y2 != y1, y2 - y1, 1e-12)
        inside ^= cross & (x < xi)
    return inside


def _lasso_around(E, members, pad=0.5):
    """A polygon around the members (their convex hull, padded) as a user would draw it."""
    from scipy.spatial import ConvexHull

    pts = E[members]
    hull = pts[ConvexHull(pts).vertices]
    c = hull.mean(0)
    return c + (hull - c) * (1 + pad / max(np.linalg.norm(hull - c, axis=1).max(), 1e-6))


@pytest.mark.parametrize("method", ["umap", "tsne", "pca"])
def test_embed_step(two_state_proj, method):
    proj = _setup(two_state_proj, method=method)
    run_cycle(proj, 0, stop="embed")
    df, states = _embedding(proj)
    assert len(df) == 32 and (proj.cycle_dir(0) / "06_embed" / "embedding.png").is_file()
    E = df[["py3staEmbed1", "py3staEmbed2"]].to_numpy()
    from py3sta.cluster import kmeans

    assert ari(kmeans(E, 2).labels, states) == 1.0, method
    # classify settings do not touch the map; msa settings do
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["classify"].update(runs={"km": {"k": [3]}}))
    assert step_state(load_project(proj.root), 0, "embed").state == "done"
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["msa"].update(n_factors=5))
    assert step_state(load_project(proj.root), 0, "embed").reasons == ["upstream msa stale"]


def test_explore_page(two_state_proj, tmp_path):
    proj = _setup(two_state_proj)
    run_cycle(proj, 0, stop="embed")
    r = CliRunner().invoke(main, ["explore", str(proj.root), "0"])
    assert r.exit_code == 0, r.output
    page = proj.cycle_dir(0) / "06_embed" / "explore.html"
    html = page.read_text()
    assert "py3sta explore" in r.output or str(page) in r.output
    data = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S).group(1))
    assert len(data["names"]) == 32 and data["cycle"] == 0
    assert set(data["axes"]) == {"06_embed (umap)", "umap_km (UMAP)", "PC1 / PC2"}
    assert data["classifications"] == ["km/K002", "umap_km/K002"]
    assert {"tomogram", "half set", "km/K002"} <= set(data["categorical"])
    assert {"PC1", "|beam z| (wedge)", "height z (px)"} <= set(data["numeric"])
    assert all(len(v) == 32 for v in data["numeric"].values())
    assert len(html) > 4e6                                               # plotly.js inlined: works offline
    assert "cdn" not in html.split('<script type="application/json"')[1].lower()
    if shutil.which("node"):                     # run the page's script on a stub DOM, simulate lasso + export
        from pathlib import Path

        harness = Path(__file__).parent / "explore_harness.js"
        res = json.loads(subprocess.run(["node", str(harness), str(page)], check=True, capture_output=True,
                                        text=True).stdout)
        assert res["first"]["names"] == ["1 (16)", "2 (16)"] and res["first"]["n"] == 32
        assert res["note"] == f'{res["moved"]} particle(s) moved from other groups to "other"'
        assert res["afterUndo"] == f"{32 - 12} particle(s) in no group (class 0)"          # undo: only "good" (12)
        exported = res["exported"]
        assert exported["format"] == "py3sta-groups" and exported["cycle"] == 0
        write = tmp_path / "exported.json"
        write.write_text(json.dumps(exported))
        lab, names = load_groups(write, data["names"])                       # what the page exports, py3sta reads
        assert names == ["good", "other"] and np.bincount(lab).tolist() == [32 - 19, 6, 13]   # good 0-6, other 6-19
        assert res["pcpick"] == "flex"
    r = CliRunner().invoke(main, ["explore", str(proj.root), "0", "--plotly-js", str(tmp_path / "none.js")])
    assert r.exit_code != 0


def test_lasso_groups_reproduce_states_through_select(two_state_proj):
    """Phase 4 acceptance: a lasso around one cluster of the map -> groups file -> classify run -> select."""
    proj = _setup(two_state_proj)
    run_cycle(proj, 0, stop="embed")
    df, states = _embedding(proj)
    E = df[["py3staEmbed1", "py3staEmbed2"]].to_numpy()
    names = df["rlnTomoParticleName"].astype(str).to_numpy()
    from py3sta.cluster import kmeans

    blob = kmeans(E, 2).labels == 1
    picked = _inside(E, _lasso_around(E, blob))
    assert (picked == blob).all()
    write_groups(proj.cycle_dir(0) / "groups" / "lasso.json", {"picked": names[picked]}, proj.root.name, 0, "06_embed")
    runs = {**RUNS, "lasso": {"method": "groups", "file": "groups/lasso.json"}}
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml",
               lambda c: (c["classify"].update(runs=runs, use="lasso"), c["select"].update(align=[], exclude=[0])))
    proj = load_project(proj.root)
    run_cycle(proj, 0, start="classify")
    out = proj.cycle_dir(0) / "05_classify"
    cs = read_star(out / "lasso" / "class_all.star")["classes"]
    assert ari(cs["py3staClass"], states) == 1.0
    classes = read_star(out / "lasso" / "classes_all.star")["classes"]
    assert classes["py3staClassName"].tolist() == ["(not in a group)", "picked"]
    agree = read_star(out / "compare.star")["compare"]
    assert "lasso" in agree["py3staRun"].tolist()
    fin = read_star(proj.cycle_dir(0) / "08_final" / "cycle-000_final.star")["particles"]
    excl = fin.set_index("rlnTomoParticleName").loc[names[~picked], "py3staExcluded"]
    assert (excl == 1).all() and fin["py3staExcluded"].sum() == (~picked).sum()
    # a new export of the same name re-runs only the lasso run
    write_groups(proj.cycle_dir(0) / "groups" / "lasso.json", {"a": names[picked][:5], "b": names[picked][5:]})
    assert step_state(load_project(proj.root), 0, "classify").state == "stale"
    run_cycle(load_project(proj.root), 0, start="classify", stop="classify")
    info = read_manifest(out)["info"]
    assert info["cached"] == ["km", "umap_km"] and info["classifications"]["lasso/all"]["k"] == 2


def test_groups_file_errors(two_state_proj, tmp_path):
    names = ["a/1", "a/2", "a/3"]
    write_groups(tmp_path / "g.json", {"x": ["a/1"], "y": ["a/3"]})
    lab, g = load_groups(tmp_path / "g.json", names)
    assert lab.tolist() == [1, 0, 2] and g == ["x", "y"]
    for groups, msg in (({"x": ["a/9"]}, "not among the active particles"), ({"x": ["a/1"], "y": ["a/1"]}, "in two groups"),
                        ({}, "no groups")):
        write_groups(tmp_path / "g.json", groups)
        with pytest.raises(ConfigError, match=msg):
            load_groups(tmp_path / "g.json", names)
    (tmp_path / "g.json").write_text('{"groups": []}')
    with pytest.raises(ConfigError, match="not a py3sta groups file"):
        load_groups(tmp_path / "g.json", names)
    cy = two_state_proj.cycle_dir(0) / "cycle.yaml"
    _edit_yaml(cy, lambda c: c["classify"].update(runs={"lasso": {"method": "groups"}}))
    with pytest.raises(ConfigError, match="needs the groups JSON file"):
        load_project(two_state_proj.root).cycle_config(0)
    _edit_yaml(cy, lambda c: c["classify"].update(runs={"lasso": {"method": "groups", "file": "groups/none.json"}}))
    with pytest.raises(FileNotFoundError, match="export it from 'py3sta explore'"):
        run_cycle(load_project(two_state_proj.root), 0, stop="classify")
    _edit_yaml(cy, lambda c: c.update(embed={"method": "pca", "factors": [1]}))
    with pytest.raises(ConfigError, match="embed.factors"):
        load_project(two_state_proj.root).cycle_config(0)
