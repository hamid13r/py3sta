import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from py3sta.geometry import euler_to_matrix
from py3sta.io.mrc import read_mrc, write_mrc
from py3sta.io.star import read_star, write_star
from py3sta.pipeline import ConfigError, init_project, load_project
from py3sta.pipeline import runner, steps
from py3sta.pipeline.manifest import StepError, check_state, read_manifest, run_step, write_manifest
from py3sta.pipeline.project import next_cycle
from py3sta.pipeline.runner import project_status, run_cycle, step_state

BOX = 16


def _object():
    ref = np.zeros((BOX,) * 3, np.float32)
    ref[8, 8, 8:13] = 3   # +x arm (long)
    ref[8, 8:11, 8] = 3   # +y arm
    ref[8:10, 8, 8] = 3   # +z stub
    return ref


def _dataset(tmp_path, n_tomos=5, n_per=8, noise_tomo="TS_04", seed=0):
    """Tomograms (apix 10) with a rotated asymmetric object at every particle; ``noise_tomo`` has none."""
    rng = np.random.default_rng(seed)
    d = tmp_path / "data"
    d.mkdir()
    ref = _object()
    zz, yy, xx = np.nonzero(ref)
    r = np.stack([xx, yy, zz], 1) - BOX // 2
    rows = []
    for k in range(n_tomos):
        name = f"TS_{k:02d}"
        vol = rng.normal(scale=0.5, size=(40, 64, 64)).astype(np.float32)
        for i in range(n_per):
            c = np.array([12 + 40 * (i % 2), 12 + 13 * (i // 2), 20.0])
            e = rng.uniform(-180, 180, 3)
            if name != noise_tomo:
                p = np.rint(r @ euler_to_matrix(*e) + c).astype(int)
                vol[p[:, 2], p[:, 1], p[:, 0]] += ref[zz, yy, xx]
            rows.append(dict(rlnCoordinateX=c[0] * 2, rlnCoordinateY=c[1] * 2, rlnCoordinateZ=c[2] * 2,
                             rlnAngleRot=e[0], rlnAngleTilt=e[1], rlnAnglePsi=e[2],
                             rlnMicrographName=f"{name}.mrc", rlnRandomSubset=1 + i % 2))
        write_mrc(d / f"{name}.mrc", vol, 10.0)
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnImagePixelSize": [5.0], "rlnImageSize": [BOX]})
    write_star({"optics": optics, "particles": pd.DataFrame(rows)}, d / "in.star")
    return d / "in.star"


@pytest.fixture
def proj(tmp_path):
    star = _dataset(tmp_path)
    p = init_project(tmp_path / "proj", star, backend="numpy", jobs=1, margin=3).project
    _edit_yaml(p.cycle_dir(0) / "cycle.yaml", lambda c: (c["reference"]["mask"].update(radius=[60.0], soft=10.0),
                                                         c["select"].update(step=30.0, bin=1),
                                                         c["align"].update(enable=False)))
    return load_project(p.root)


def _edit_yaml(path, fn):
    d = yaml.safe_load(path.read_text())
    fn(d)
    path.write_text(yaml.safe_dump(d, sort_keys=False))


def test_average_step(proj):
    run_cycle(proj, 0, log=print)
    st = step_state(proj, 0, "average")
    assert st.state == "done", st.describe()
    f = proj.cycle_dir(0) / "01_average"
    for name in ("average.mrc", "half1.mrc", "half2.mrc", "fsc.txt", "per_tomo.csv", "manifest.json"):
        assert (f / name).is_file()
    m = read_manifest(f)
    assert m["info"]["n_used"] == 40 and m["info"]["low_cc_tomograms"] == ["TS_04"]
    assert m["git"] is None or len(m["git"]) >= 7
    cc = pd.read_csv(f / "per_tomo.csv").set_index("tomo")
    assert cc.loc["TS_04", "cc"] < 0.3 < cc.drop("TS_04")["cc"].min()
    # the average from the stack recovers the object
    avg, _ = read_mrc(f / "average.mrc")
    assert np.corrcoef(avg.ravel(), _object().ravel())[0, 1] > 0.5


def test_staleness_reasons(proj):
    run_cycle(proj, 0)
    cin = proj.cycle_input(0)
    os.utime(cin)                                                   # touch: not an edit
    assert step_state(proj, 0, "average").state == "done"
    _edit_yaml(proj.root / "project.yaml", lambda c: c["compute"].update(backend="torch", gpu="cpu"))
    assert step_state(load_project(proj.root), 0, "average").state == "done"   # compute is not a parameter
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["average"].update(normalize=False))
    st = step_state(proj, 0, "average")
    assert st.state == "stale" and "average.normalize True→False" in st.reasons
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["average"].update(normalize=True))
    assert step_state(proj, 0, "average").state == "done"
    b = read_star(cin)
    b["particles"].loc[0, "rlnAngleRot"] += 10
    write_star(b, cin)
    st = step_state(proj, 0, "average")
    assert st.state == "stale" and st.reasons == ["cycle-000/00_input.star changed"]


def test_replaced_tomogram(proj):
    run_cycle(proj, 0)
    t = proj.root.parent / "data" / "TS_01.mrc"
    vol, apix = read_mrc(t, mmap=False)
    write_mrc(t, vol[:, :, ::-1].copy(), apix)
    st = step_state(proj, 0, "average")
    assert st.state == "stale" and st.reasons == ["1 tomogram(s) changed: TS_01.mrc"]
    logs = []
    run_cycle(proj, 0, log=logs.append)
    assert any("8 re-cropped (tomogram changed: TS_01)" in m for m in logs)
    assert step_state(proj, 0, "average").state == "done"


def test_excluded_particles_skipped(proj):
    cin = proj.cycle_input(0)
    b = read_star(cin)
    b["particles"]["py3staExcluded"] = [1] * 8 + [0] * 32           # all of TS_00
    write_star(b, cin)
    run_cycle(proj, 0)
    m = read_manifest(proj.cycle_dir(0) / "01_average")
    assert (m["info"]["n_used"], m["info"]["n_excluded"]) == (32, 8)
    assert "TS_00" not in pd.read_csv(proj.cycle_dir(0) / "01_average" / "per_tomo.csv")["tomo"].tolist()


def test_torch_backend_matches_numpy(proj):
    pytest.importorskip("torch")
    run_cycle(proj, 0)
    a, _ = read_mrc(proj.cycle_dir(0) / "01_average" / "average.mrc", mmap=False)
    _edit_yaml(proj.root / "project.yaml", lambda c: c["compute"].update(backend="torch", gpu="cpu"))
    run_cycle(load_project(proj.root), 0)
    b, _ = read_mrc(proj.cycle_dir(0) / "01_average" / "average.mrc", mmap=False)
    np.testing.assert_allclose(b, a, atol=1e-4)


# ------------------------------------------------------------ run_step / manifests
def _star_step(text):
    def fn(ctx):
        ctx.output_star("out.star").write_text(text)
    return fn


def test_edited_output_protected(tmp_path):
    f = tmp_path / "step"
    run_step(tmp_path, f, "x", {}, _star_step("data_\n"))
    assert check_state(tmp_path, f, {}).edited == []
    (f / "out.star").write_text("data_\n# my edit\n")
    st = check_state(tmp_path, f, {})
    assert st.state == "done" and st.edited == ["out.star"]
    with pytest.raises(StepError, match="edited by hand"):
        run_step(tmp_path, f, "x", {}, _star_step("data_\n"))
    run_step(tmp_path, f, "x", {}, _star_step("data_\n"), force=True)
    backups = list(f.glob("out.edited-*.star"))
    assert len(backups) == 1 and "my edit" in backups[0].read_text()
    assert check_state(tmp_path, f, {}).edited == []


def test_failed_and_changed_during_run(tmp_path):
    f = tmp_path / "step"

    def boom(ctx):
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        run_step(tmp_path, f, "x", {}, boom)
    st = check_state(tmp_path, f, {})
    assert st.state == "failed" and st.reasons == ["ValueError: bad input"]

    tomo = tmp_path / "t.mrc"
    tomo.write_bytes(b"0" * 10)

    def swap(ctx):
        ctx.use_files([tomo], "tomogram")
        tomo.write_bytes(b"1" * 20)

    with pytest.raises(StepError, match="changed during the run"):
        run_step(tmp_path, f, "x", {}, swap)
    assert check_state(tmp_path, f, {}).state == "failed"


def test_interrupted(tmp_path):
    f = tmp_path / "step"
    run_step(tmp_path, f, "x", {}, lambda ctx: None)
    m = read_manifest(f)
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    m.update(status="running", pid=p.pid)
    write_manifest(f, m)
    assert check_state(tmp_path, f, {}).state == "interrupted"
    m.update(pid=os.getpid())
    write_manifest(f, m)
    assert check_state(tmp_path, f, {}).state == "running"
    with pytest.raises(StepError, match="already running"):
        run_step(tmp_path, f, "x", {}, lambda ctx: None)


def test_upstream_rules(proj, monkeypatch):
    """A fake step after 'average': stale upstream refuses (option a), --force overrides,
    re-running upstream makes it stale."""
    ran = []
    fake = steps.Step("reference", "02_reference", ("average",), lambda p, c: {}, lambda ctx, *a: ran.append(1))
    new = tuple(fake if s.name == "reference" else s for s in steps.STEPS)
    monkeypatch.setattr(steps, "STEPS", new)
    monkeypatch.setattr(runner, "STEPS", new)
    monkeypatch.setitem(steps.BY_NAME, "reference", fake)

    with pytest.raises(StepError, match="upstream step average is missing"):
        run_cycle(proj, 0, start="reference")
    run_cycle(proj, 0, stop="reference")                             # average + reference
    assert ran == [1] and step_state(proj, 0, "reference").state == "done"
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["average"].update(halves=False))
    st = step_state(proj, 0, "reference")
    assert st.state == "stale" and st.reasons == ["upstream average stale"]
    with pytest.raises(StepError, match="run with --from average"):
        run_cycle(proj, 0, start="reference")
    run_cycle(proj, 0, start="reference", stop="reference", force=True)
    run_cycle(proj, 0, stop="average")                               # re-run upstream only
    st = step_state(proj, 0, "reference")
    assert st.state == "stale" and st.reasons == ["upstream average re-run"]


def test_status_and_cli(proj):
    from click.testing import CliRunner

    from py3sta.cli import main

    r = CliRunner().invoke(main, ["status", str(proj.root)])
    assert r.exit_code == 0, r.output
    assert "stack: not built yet" in r.output and "01_average     missing" in r.output
    r = CliRunner().invoke(main, ["run", str(proj.root), "cycle-000"])
    assert r.exit_code == 0, r.output
    assert "stack built: 40 new" in r.output
    r = CliRunner().invoke(main, ["status", str(proj.root)])
    assert "stack: 40 particles" in r.output and "low CC: TS_04" in r.output
    st = project_status(proj)
    assert st.drift_warning == 0 and st.cycles[0].n_particles == 40
    r = CliRunner().invoke(main, ["run", str(proj.root), "0", "--from", "reference", "--to", "reference"])
    assert r.exit_code == 0 and "1 reference(s)" in r.output, r.output
    r = CliRunner().invoke(main, ["run", str(proj.root), "0", "--from", "msa", "--to", "msa"])
    assert r.exit_code != 0 and "no implemented step" in r.output
    r = CliRunner().invoke(main, ["run", str(proj.root), "0", "--from", "final"])
    assert r.exit_code != 0 and "upstream step select is stale" in r.output and "upstream reference re-run" in r.output
    r = CliRunner().invoke(main, ["run", str(proj.root), "0", "--from", "align"])
    assert r.exit_code == 0 and "40 particles, poses from select" in r.output, r.output
    assert "align disabled: poses passed through" in r.output
    r = CliRunner().invoke(main, ["next", str(proj.root), "cycle-000"])
    assert r.exit_code == 0 and "cycle-001: 40 particles from cycle-000_final.star" in r.output, r.output
    r = CliRunner().invoke(main, ["status", str(proj.root)])
    assert "cycle-001  40 particles" in r.output and "08_final       done" in r.output, r.output


# ------------------------------------------------------------ final / next
def _final(proj, n=0):
    return proj.cycle_dir(n) / "08_final" / f"cycle-{n:03d}_final.star"


def test_final_and_next(proj):
    cin = proj.cycle_input(0)
    b = read_star(cin)
    b["particles"]["py3staExcluded"] = [1] * 8 + [0] * 32
    write_star(b, cin)
    with pytest.raises(ConfigError, match="final step is missing"):
        next_cycle(proj, 0)
    run_cycle(proj, 0)
    fin = read_star(_final(proj))["particles"]
    assert len(fin) == 40 and (fin["py3staCycle"] == 0).all() and fin["py3staExcluded"].sum() == 8
    assert set(read_star(_final(proj))) == set(b)                       # optics block kept
    rep = next_cycle(proj, 0)
    assert (rep.n, rep.n_particles, rep.warnings) == (1, 40, [])
    assert proj.cycles == [0, 1] and proj.cycle_config(1).cycle == 1
    text = (proj.cycle_dir(1) / "cycle.yaml").read_text()
    assert text == (proj.cycle_dir(0) / "cycle.yaml").read_text().replace("cycle: 0", "cycle: 1")   # comments kept
    assert (proj.cycle_input(1)).read_bytes() == _final(proj).read_bytes()
    with pytest.raises(ConfigError, match="exists"):
        next_cycle(proj, 0)
    # cycle 1 runs from the copy; excluded particles stay excluded
    run_cycle(proj, 1, stop="average")
    assert read_manifest(proj.cycle_dir(1) / "01_average")["info"]["n_used"] == 32


def test_next_refuses_stale_final_and_keeps_edits(proj):
    run_cycle(proj, 0)
    fin = _final(proj)
    b = read_star(fin)
    b["particles"].loc[3, "rlnAngleRot"] += 5                         # hand edit of the final STAR
    write_star(b, fin)
    rep = next_cycle(proj, 0)
    assert rep.warnings and "edited by hand" in rep.warnings[0]
    assert read_star(proj.cycle_input(1))["particles"].loc[3, "rlnAngleRot"] == b["particles"].loc[3, "rlnAngleRot"]
    cin = proj.cycle_input(0)
    b = read_star(cin)
    b["particles"].loc[0, "rlnAngleTilt"] += 1
    write_star(b, cin)
    assert step_state(proj, 0, "final").state == "stale"
    with pytest.raises(ConfigError, match="final step is stale"):
        next_cycle(proj, 0, force=False)
    rep = next_cycle(proj, 0, force=True)
    assert any("stale" in w for w in rep.warnings)


def test_final_reads_pose_source(proj, monkeypatch):
    """With a pose-writing step implemented, final copies its poses, depends on it and checks no particle is lost."""
    from py3sta.io.particles import ParticleSet
    from py3sta.poses import apply_delta

    def fake_align(ctx, project, n, c):
        ps = ParticleSet.from_star(project.cycle_input(n))
        moved = apply_delta(ps, euler_to_matrix(10, 20, 30), np.array([1.0, 0, 0]), project.config.apix)
        b = read_star(project.cycle_input(n))
        cols = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
        b["particles"][cols] = moved.df[cols].to_numpy()
        write_star(b, ctx.output_star("aligned.star"))

    fake = steps.Step("align", "03_align", ("reference",), lambda p, c: {}, fake_align, poses="aligned.star")
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["select"].update(align=[]))   # leave: poses kept
    new = tuple(fake if s.name == "align" else s for s in steps.STEPS)
    monkeypatch.setattr(steps, "STEPS", new)
    monkeypatch.setattr(runner, "STEPS", new)
    monkeypatch.setitem(steps.BY_NAME, "align", fake)
    run_cycle(proj, 0)
    m = read_manifest(proj.cycle_dir(0) / "07_select")
    assert m["info"]["pose_source"] == "align" and "align" in m["inputs"]["steps"]
    assert read_manifest(_final(proj).parent)["info"]["pose_source"] == "select"
    fin = read_star(_final(proj))["particles"]
    aligned = read_star(proj.cycle_dir(0) / "03_align" / "aligned.star")["particles"]
    np.testing.assert_allclose(fin["rlnAngleRot"], aligned["rlnAngleRot"])
    run_cycle(proj, 0, start="align", stop="align")
    assert "upstream align re-run" in step_state(proj, 0, "select").reasons
    assert step_state(proj, 0, "final").reasons == ["upstream select stale"]
    a = proj.cycle_dir(0) / "03_align" / "aligned.star"
    b = read_star(a)
    b["particles"] = b["particles"].iloc[:-1]
    write_star(b, a)
    with pytest.raises(ValueError, match="1 particle"):
        run_cycle(proj, 0, start="classify")
