import numpy as np
import pytest
from click.testing import CliRunner

from py3sta.cli import main
from py3sta.io.star import read_star, write_star
from py3sta.pipeline import ConfigError
from py3sta.pipeline.flags import flag_particles
from py3sta.pipeline.manifest import read_manifest
from py3sta.pipeline.project import next_cycle
from py3sta.pipeline.runner import run_cycle, step_state

from test_pipeline_run import _edit_yaml, proj  # noqa: F401  (fixture)


def _flags(proj, n=0):
    df = read_star(proj.cycle_input(n))["particles"]
    ex = df["py3staExcluded"].to_numpy() if "py3staExcluded" in df else np.zeros(len(df), int)
    return df, ex


def _classified(proj):
    """Cycle 0 run with classes = rlnRandomSubset (poses kept), cycle 1 created."""
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml",
               lambda c: (c["classify"].update(column="rlnRandomSubset"), c["select"].update(align=[])))
    run_cycle(proj, 0)
    next_cycle(proj, 0)


def test_tomo_exclude_include_roundtrip(proj):
    rep = flag_particles(proj, 0, exclude=["tomo:TS_01,TS_03"], dry_run=True)
    assert (rep.newly_excluded, rep.written) == (16, False)
    assert "py3staExcluded" not in read_star(proj.cycle_input(0))["particles"]
    rep = flag_particles(proj, 0, exclude=["tomo:TS_01,TS_03"])
    df, ex = _flags(proj)
    assert rep.written and ex.sum() == 16 and set(df.loc[ex == 1, "rlnMicrographName"].str.split("/").str[-1]) == {"TS_01.mrc", "TS_03.mrc"}
    assert (df.loc[ex == 1, "py3staExcludedCycle"] == 0).all() and (df.loc[ex == 0, "py3staExcludedCycle"] == -1).all()
    rep = flag_particles(proj, 0, exclude=["tomo:TS_01"], include=["tomo:TS_03"])
    assert (rep.newly_excluded, rep.newly_included, rep.unchanged, rep.excluded_after) == (0, 8, 8, 8)
    rep = flag_particles(proj, 0, include=["all"])
    assert rep.excluded_after == 0 and _flags(proj)[1].sum() == 0
    log = (proj.cycle_dir(0) / "flag.log").read_text().splitlines()
    assert len(log) == 3 and "+16 excluded" in log[0]


def test_errors(proj, tmp_path):
    with pytest.raises(ConfigError, match="unknown tomogram"):
        flag_particles(proj, 0, exclude=["tomo:TS_99"])
    with pytest.raises(ConfigError, match="selection must be one of"):
        flag_particles(proj, 0, exclude=["tomos:TS_01"])
    with pytest.raises(ConfigError, match="by both --exclude and --include"):
        flag_particles(proj, 0, exclude=["tomo:TS_01"], include=["all"])
    with pytest.raises(ConfigError, match="no final or selected STAR"):
        flag_particles(proj, 0, exclude=["class:0:1"])
    with pytest.raises(ConfigError, match="no cycle-005"):
        flag_particles(proj, 0, include=["excluded:5"])
    (tmp_path / "names.txt").write_text("TS_00/1\nTS_00/99\n")
    with pytest.raises(ConfigError, match="1 particle name\\(s\\) not in this cycle: TS_00/99"):
        flag_particles(proj, 0, exclude=[f"file:{tmp_path / 'names.txt'}"])
    with pytest.raises(ConfigError, match="nothing to do"):
        flag_particles(proj, 0)
    assert not (proj.cycle_dir(0) / "flag.log").exists()


def test_file_selection(proj, tmp_path):
    (tmp_path / "names.txt").write_text("# bad picks\nTS_00/1\nTS_02/3\n")
    flag_particles(proj, 0, exclude=[f"file:{tmp_path / 'names.txt'}"])
    df, ex = _flags(proj)
    assert sorted(df.loc[ex == 1, "rlnTomoParticleName"]) == ["TS_00/1", "TS_02/3"]
    b = read_star(proj.cycle_input(0))
    b["particles"] = b["particles"].iloc[:5]
    write_star(b, tmp_path / "sub.star")
    rep = flag_particles(proj, 0, exclude=[f"file:{tmp_path / 'sub.star'}"])
    assert (rep.newly_excluded, rep.unchanged) == (4, 1)


def test_class_and_excluded_selections(proj):
    _classified(proj)
    with pytest.raises(ConfigError, match="no class \\[3\\] in cycle-000 \\(classes: 1, 2\\)"):
        flag_particles(proj, 1, exclude=["class:0:3"])
    rep = flag_particles(proj, 1, exclude=["class:cycle-000:2"])
    df, ex = _flags(proj, 1)
    assert rep.newly_excluded == 20 and (df.loc[ex == 1, "rlnRandomSubset"] == 2).all()
    # class 2 excluded by flag in cycle 1, class 1 by select in cycle 2; excluded:1 undoes cycle 1 only
    run_cycle(proj, 1)
    next_cycle(proj, 1)
    _edit_yaml(proj.cycle_dir(2) / "cycle.yaml", lambda c: c["select"].update(exclude=[1]))
    run_cycle(proj, 2)
    nxt = next_cycle(proj, 2)
    df, ex = _flags(proj, nxt.n)
    assert ex.sum() == 40 and set(df.loc[df["rlnRandomSubset"] == 2, "py3staExcludedCycle"]) == {1}
    rep = flag_particles(proj, nxt.n, include=["excluded:1"])
    assert rep.newly_included == 20 and rep.excluded_after == 20


def test_steps_stale_and_cli(proj):
    run_cycle(proj, 0)
    r = CliRunner().invoke(main, ["flag", str(proj.root), "0", "--exclude", "tomo:TS_04", "--dry-run"])
    assert r.exit_code == 0 and "dry run: would change: +8 excluded" in r.output, r.output
    assert step_state(proj, 0, "average").state == "done"
    r = CliRunner().invoke(main, ["flag", str(proj.root), "cycle-000", "--exclude", "tomo:TS_04"])
    assert r.exit_code == 0 and "excluded 0 -> 8 of 40; active 32" in r.output, r.output
    assert "steps now stale: average, reference, classify, select, final" in r.output
    st = step_state(proj, 0, "average")
    assert st.state == "stale" and st.reasons == ["cycle-000/00_input.star changed"]
    run_cycle(proj, 0)
    assert read_manifest(proj.cycle_dir(0) / "01_average")["info"]["n_used"] == 32
    r = CliRunner().invoke(main, ["flag", str(proj.root), "0", "--exclude", "tomo:nope"])
    assert r.exit_code != 0 and "unknown tomogram" in r.output
