import os

import numpy as np
import pandas as pd
import pytest
import yaml

from py3sta.io.mrc import write_mrc
from py3sta.io.star import read_star, write_star
from py3sta.pipeline import ConfigError, CycleConfig, ProjectConfig, init_project, load_project
from py3sta.pipeline.config import from_dict, read_cycle_config, read_project_config, write_yaml

XML = '<?xml version="1.0" encoding="utf-16"?>\n<TiltSeries AreAnglesInverted="True">\n<Angles>-40\n0\n50</Angles>\n</TiltSeries>'


def _project_cfg(**kw):
    return ProjectConfig(**{"box": 32, "apix": 10.0, "ts_apix": 5.0, **kw})


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data))
    return p


def test_defaults_roundtrip(tmp_path):
    p = _project_cfg()
    write_yaml(p, tmp_path / "project.yaml")
    assert read_project_config(tmp_path / "project.yaml") == p
    c = CycleConfig.default_for(p)
    write_yaml(c, tmp_path / "cycle.yaml")
    assert read_cycle_config(tmp_path / "cycle.yaml", p) == c
    assert c.reference.mask.radius == [128.0] and c.reference.mask.soft == 32.0   # 40 % / 10 % of 320 Å


def test_unknown_key_suggests(tmp_path):
    f = _write(tmp_path, "cycle.yaml", {"reference": {"lowpas": 20}})
    with pytest.raises(ConfigError, match=r"reference\.lowpas: unknown key \(did you mean reference\.lowpass\?\)"):
        read_cycle_config(f)


def test_required_and_types(tmp_path):
    with pytest.raises(ConfigError, match="box: required"):
        read_project_config(_write(tmp_path, "a.yaml", {"apix": 1.0, "ts_apix": 1.0}))
    with pytest.raises(ConfigError, match="box: expected int, got 32.5"):
        read_project_config(_write(tmp_path, "b.yaml", {"box": 32.5, "apix": 1.0, "ts_apix": 1.0}))
    with pytest.raises(ConfigError, match=r"wedge\.use: expected bool"):
        from_dict(ProjectConfig, {"box": 32, "apix": 1.0, "ts_apix": 1.0, "wedge": {"use": "yes"}}, "x")
    c = from_dict(CycleConfig, {"reference": {"lowpass": None, "mask": {"radius": 100}}}, "x")
    assert c.reference.lowpass is None and c.reference.mask.radius == [100.0]   # int -> float, scalar -> list
    with pytest.raises(ConfigError, match=r"compute\.gpu: expected int or str"):
        from_dict(ProjectConfig, {"box": 32, "apix": 1.0, "ts_apix": 1.0, "compute": {"gpu": 1.5}}, "x")


def test_value_checks_collect_all_errors():
    p = _project_cfg(box=4)
    p.stack.margin = 0
    p.compute.gpu = -1
    p.particle_radius = 500.0
    with pytest.raises(ConfigError) as e:
        p.validate()
    msg = str(e.value)
    assert "box: must be >= 8" in msg and "stack.margin" in msg and "compute.gpu" in msg and "particle_radius" in msg
    c = CycleConfig()
    c.ctf.mode = "phaseflip"
    with pytest.raises(ConfigError, match="ctf.mode: must be one of"):
        c.validate()


def test_cycle_checks_against_project():
    p = _project_cfg()                         # box 320 Å
    c = CycleConfig.default_for(p)
    c.reference.mask.radius = [150.0]          # + soft 32 > 160
    c.reference.lowpass = 15.0                 # Nyquist 20 Å
    c.reference.highpass = 10.0
    with pytest.raises(ConfigError) as e:
        c.validate(p)
    msg = str(e.value)
    assert "exceeds half the box" in msg and "beyond Nyquist" in msg and "larger than lowpass" in msg
    c = CycleConfig.default_for(p)
    c.reference.mask.shape = "ellipsoid"
    with pytest.raises(ConfigError, match="needs 3 value"):
        c.validate(p)
    c.reference.mode = "file"
    c.reference.mask.radius = [100.0, 80.0, 60.0]
    with pytest.raises(ConfigError, match="reference.files: mode 'file' needs"):
        c.validate(p)


def test_ctf_needs_wedge():
    p = _project_cfg()
    p.wedge.use = False
    c = CycleConfig.default_for(p)
    c.ctf.mode = "flip"
    c.validate()                                # fine on its own
    with pytest.raises(ConfigError, match="needs tilt information"):
        c.validate(p)


def test_norm_radius():
    p = _project_cfg()
    assert p.norm_radius_vox == 16.0
    p.particle_radius = 100.0
    assert p.norm_radius_vox == 10.0


# ---------------------------------------------------------------- init
def _dataset(tmp_path, xml=True, apix=(10.0, 10.0)):
    """Two 32^3 tomograms + Warp XMLs + a STAR file with paths relative to the STAR directory."""
    d = tmp_path / "data"
    (d / "tomo").mkdir(parents=True)
    rows = []
    for k, (name, a) in enumerate(zip(["TS_01", "TS_02"], apix)):
        write_mrc(d / "tomo" / f"{name}.mrc_10.00Apx.mrc", np.zeros((32, 32, 32), np.float32), a)
        if xml:
            (d / "tomo" / f"{name}.mrc.xml").write_bytes(XML.encode("utf-16"))
        for i in range(3 + k):
            r = {"rlnCoordinateX": 30.0 + i, "rlnCoordinateY": 30.0, "rlnCoordinateZ": 30.0,
                 "rlnAngleRot": 10.0 * i, "rlnAngleTilt": 20.0, "rlnAnglePsi": 0.0,
                 "rlnMicrographName": f"tomo/{name}.mrc_10.00Apx.mrc", "rlnRandomSubset": 1 + i % 2}
            if xml:
                r["rlnWarpXMLfile"] = f"tomo/{name}.mrc.xml"
            rows.append(r)
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnImagePixelSize": [5.0], "rlnImageSize": [16]})
    write_star({"optics": optics, "particles": pd.DataFrame(rows)}, d / "in.star")
    return d / "in.star"


def test_init_project(tmp_path, monkeypatch):
    star = _dataset(tmp_path)
    monkeypatch.chdir(tmp_path)                # paths must resolve via the STAR directory, not cwd
    rep = init_project(tmp_path / "proj", star)
    assert rep.n_particles == 7 and rep.per_tomo == {"TS_01": 3, "TS_02": 4}
    assert rep.tilt_ranges["TS_01"] == (-50.0, 40.0)            # AreAnglesInverted -> sign -1
    proj = load_project(tmp_path / "proj")
    cfg = proj.config
    assert (cfg.box, cfg.apix, cfg.ts_apix, cfg.wedge.use) == (16, 10.0, 5.0, True)
    assert proj.cycle_config(0).ctf.mode == "none"
    assert proj.cycles == [0] and proj.cycle_config(0).cycle == 0
    b = read_star(proj.input_star)
    assert list(b) == ["optics", "particles"]
    df = b["particles"]
    for col in ("rlnMicrographName", "rlnWarpXMLfile"):
        assert all(os.path.isabs(p) and os.path.isfile(p) for p in df[col])
    assert list(df["rlnAngleRot"]) == [0, 10, 20, 0, 10, 20, 30]  # everything else untouched
    assert list(df["rlnTomoParticleName"]) == ["TS_01/1", "TS_01/2", "TS_01/3", "TS_02/1", "TS_02/2", "TS_02/3", "TS_02/4"]
    cin = proj.cycle_input(0)
    assert not cin.is_symlink() and read_star(cin)["particles"].equals(df)   # a copy


def test_init_refuses_overwrite_and_force(tmp_path):
    star = _dataset(tmp_path)
    init_project(tmp_path / "proj", star)
    with pytest.raises(ConfigError, match="exists"):
        init_project(tmp_path / "proj", star)
    rep = init_project(tmp_path / "proj", star, box=24, force=True)
    assert load_project(tmp_path / "proj").config.box == rep.project.config.box == 24


def test_init_without_xml(tmp_path):
    star = _dataset(tmp_path, xml=False)
    rep = init_project(tmp_path / "proj", star)
    assert rep.project.config.wedge.use is False and any("wedge" in w for w in rep.warnings)
    with pytest.raises(ConfigError, match="rlnWarpXMLfile"):
        init_project(tmp_path / "p2", star, wedge=True)
    with pytest.raises(ConfigError, match="needs tilt information"):
        init_project(tmp_path / "p3", star, ctf_mode="flip")


def test_load_cycle_input_checks_edits(tmp_path):
    from py3sta.pipeline.project import load_cycle_input

    star = _dataset(tmp_path)
    proj = init_project(tmp_path / "proj", star).project
    cin = load_cycle_input(proj, 0)
    assert (len(cin.particles), cin.n_excluded, sorted(cin.tomograms)) == (7, 0, ["TS_01", "TS_02"])
    b = read_star(proj.cycle_input(0))
    b["particles"]["py3staExcluded"] = [1, 1, 1, 0, 0, 0, 0]       # all of TS_01 excluded
    write_star(b, proj.cycle_input(0))
    cin = load_cycle_input(proj, 0)
    assert (len(cin.particles), cin.n_excluded, sorted(cin.tomograms)) == (4, 3, ["TS_02"])
    b["particles"].loc[1, "rlnTomoParticleName"] = "TS_01/1"
    write_star(b, proj.cycle_input(0))
    with pytest.raises(ConfigError, match="not unique"):
        load_cycle_input(proj, 0)
    b["particles"].loc[1, "rlnTomoParticleName"] = "TS_01/2"
    b["particles"]["rlnMicrographName"] = "/nonexistent/TS_01.mrc"
    write_star(b, proj.cycle_input(0))
    with pytest.raises(FileNotFoundError):
        load_cycle_input(proj, 0)


def test_init_errors(tmp_path):
    star = _dataset(tmp_path, apix=(10.0, 20.0))
    with pytest.raises(ConfigError, match="differ in voxel size"):
        init_project(tmp_path / "proj", star)
    (tmp_path / "data" / "tomo" / "TS_02.mrc_10.00Apx.mrc").unlink()
    with pytest.raises(FileNotFoundError, match="TS_02"):
        init_project(tmp_path / "proj", star)
    assert not (tmp_path / "proj").exists()    # nothing written on failure


def test_cycle_number_must_match_folder(tmp_path):
    star = _dataset(tmp_path)
    proj = init_project(tmp_path / "proj", star).project
    f = proj.cycle_dir(0) / "cycle.yaml"
    f.write_text(f.read_text().replace("cycle: 0", "cycle: 3"))
    with pytest.raises(ConfigError, match="does not match the folder cycle-000"):
        proj.cycle_config(0)


def test_cli_init(tmp_path):
    from click.testing import CliRunner

    from py3sta.cli import main

    star = _dataset(tmp_path)
    r = CliRunner().invoke(main, ["init", str(tmp_path / "proj"), "--star", str(star), "--backend", "torch",
                                  "--gpu", "cpu"])
    assert r.exit_code == 0, r.output
    assert "7 particles in 2 tomograms" in r.output
    assert load_project(tmp_path / "proj").config.compute.gpu == "cpu"
    r = CliRunner().invoke(main, ["init", str(tmp_path / "proj"), "--star", str(star)])
    assert r.exit_code != 0 and "exists" in r.output
