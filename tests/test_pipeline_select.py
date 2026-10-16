import numpy as np
import pytest

from py3sta.geometry import euler_to_matrix
from py3sta.io.mrc import read_mrc, write_mrc
from py3sta.io.particles import ParticleSet
from py3sta.io.star import read_star, write_star
from py3sta.pipeline import ConfigError, load_project
from py3sta.pipeline.manifest import StepError, read_manifest
from py3sta.pipeline.project import next_cycle
from py3sta.pipeline.runner import run_cycle, step_state
from py3sta.poses import apply_delta, invert

from test_align import APIX, BOX, chiral_object
from test_pipeline_run import _edit_yaml, _final, proj  # noqa: F401  (fixture)

TOMO = 96


@pytest.fixture
def chiral_proj(tmp_path):
    """2 tomograms (96^3, 4 Å) with 8 copies each of the chiral object at known poses, analytic (no
    interpolation), light noise; box 40. rlnRandomSubset alternates."""
    import pandas as pd

    from py3sta.pipeline import init_project

    rng = np.random.default_rng(7)
    d = tmp_path / "data"
    d.mkdir()
    rows = []
    grid = np.array([[x, y, z] for x in (24, 72) for y in (24, 72) for z in (24, 72)], float)
    for k in range(2):
        name = f"TS_{k:02d}"
        vol = rng.normal(scale=0.05, size=(TOMO,) * 3).astype(np.float32)
        for i, c in enumerate(grid + rng.uniform(-0.5, 0.5, grid.shape)):
            e = rng.uniform(-180, 180, 3)
            vol += chiral_object(TOMO, APIX, shift_A=(c - TOMO // 2) * APIX, M=euler_to_matrix(*e))
            rows.append(dict(rlnCoordinateX=c[0], rlnCoordinateY=c[1], rlnCoordinateZ=c[2], rlnAngleRot=e[0],
                             rlnAngleTilt=e[1], rlnAnglePsi=e[2], rlnMicrographName=f"{name}.mrc",
                             rlnRandomSubset=1 + i % 2))
        write_mrc(d / f"{name}.mrc", vol, APIX)
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnImagePixelSize": [APIX], "rlnImageSize": [BOX]})
    write_star({"optics": optics, "particles": pd.DataFrame(rows)}, d / "in.star")
    p = init_project(tmp_path / "proj", d / "in.star", backend="numpy", jobs=1, margin=3).project
    _edit_yaml(p.cycle_dir(0) / "cycle.yaml",
               lambda c: (c["reference"]["mask"].update(radius=[66.0], soft=12.0),
                          c["select"].update(step=30.0, bin=1, lowpass=10.0, max_shift=24.0),
                          c["align"].update(enable=False),
                          c["msa"].update(enable=False)))
    return load_project(p.root)

R2, T2 = euler_to_matrix(20.0, 15.0, -10.0), np.array([1.0, 0.0, -1.0])        # class 2 pose error (voxels)


def _angles(a: ParticleSet, b: ParticleSet) -> np.ndarray:
    tr = np.einsum("nij,nij->n", a.matrices, b.matrices)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def _perturb_class2(proj):
    """Classes from rlnRandomSubset; class 2 poses moved by the inverse of (R2, T2). Returns the true poses."""
    cin = proj.cycle_input(0)
    true = ParticleSet.from_star(cin)
    b = read_star(cin)
    wrong = apply_delta(true, *invert(R2, T2), proj.config.apix, rows=true.df["rlnRandomSubset"].to_numpy() == 2)
    cols = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
    for c in cols:
        b["particles"][c] = wrong.df[c].to_numpy()
    write_star(b, cin)
    write_mrc(proj.root / "target.mrc", chiral_object(), APIX)
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml",
               lambda c: (c["classify"].update(column="rlnRandomSubset"), c["select"].update(target="target.mrc")))
    return true


def test_class_transforms_restore_poses(chiral_proj):
    proj = chiral_proj
    true = _perturb_class2(proj)
    run_cycle(proj, 0)
    cl = proj.cycle_dir(0) / "05_classify"
    assert read_manifest(cl)["info"]["counts"] == {"1": 8, "2": 8}
    assert read_mrc(cl / "classes.mrcs")[0].shape == (2 * BOX, BOX, BOX)
    tr = read_star(proj.cycle_dir(0) / "07_select" / "transforms.star")["transforms"].set_index("py3staClass")
    assert list(tr["py3staClassAction"]) == ["align", "align"]
    fin = ParticleSet.from_star(_final(proj))
    err = _angles(fin, true)
    assert err.max() < 1.0, err
    shift = np.abs(fin.centers_px - true.centers_px).max() * true.ts_apix / proj.config.apix   # voxels
    assert shift < 0.3, shift
    df = read_star(_final(proj))["particles"]
    assert (df["rlnClassNumber"] == df["rlnRandomSubset"]).all() and df["py3staClassCC"].notna().all()


def test_leave_and_exclude(chiral_proj):
    proj = chiral_proj
    _perturb_class2(proj)
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["select"].update(align=[], leave=[2], exclude=[1]))
    before = ParticleSet.from_star(proj.cycle_input(0))
    run_cycle(proj, 0)
    df = read_star(_final(proj))["particles"]
    one = df["rlnRandomSubset"] == 1
    assert (df.loc[one, "py3staExcluded"] == 1).all() and (df.loc[~one, "py3staExcluded"] == 0).all()
    assert (df.loc[one, "py3staExcludedCycle"] == 0).all()
    np.testing.assert_allclose(_angles(ParticleSet.from_star(_final(proj)), before), 0, atol=1e-4)   # all poses kept
    next_cycle(proj, 0)
    run_cycle(proj, 1, stop="average")
    assert read_manifest(proj.cycle_dir(1) / "01_average")["info"]["n_used"] == 8


def test_classify_tracking_and_validation(proj):
    run_cycle(proj, 0)
    assert "no column rlnClassNumber" in read_manifest(proj.cycle_dir(0) / "05_classify")["warnings"][0]
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["classify"].update(column="rlnRandomSubset"))
    p = load_project(proj.root)
    st = step_state(p, 0, "classify")
    assert st.state == "stale" and st.reasons == ["classify.column rlnClassNumber→rlnRandomSubset"]
    assert step_state(p, 0, "final").reasons == ["upstream select stale"]
    sel = p.cycle_dir(0) / "07_select" / "selected.star"
    b = read_star(sel)
    b["particles"].loc[0, "rlnAngleRot"] += 3
    write_star(b, sel)
    with pytest.raises(StepError, match="edited by hand"):
        run_cycle(p, 0, start="classify")
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["select"].update(leave=[1], exclude=[1]))
    with pytest.raises(ConfigError, match="select.exclude: class\\(es\\) \\[1\\] also in select.leave"):
        load_project(proj.root).cycle_config(0)
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["select"].update(leave=[], exclude=[], lowpass=25.0, bin=2))
    with pytest.raises(ConfigError, match="beyond Nyquist of the binned search"):
        load_project(proj.root).cycle_config(0)


def test_missing_class_column_is_one_class(proj):
    b = read_star(proj.cycle_input(0))
    b["particles"]["rlnClassNumber"] = 1.5
    write_star(b, proj.cycle_input(0))
    with pytest.raises(ValueError, match="not an integer for 40"):
        run_cycle(proj, 0, stop="classify")
    b["particles"] = b["particles"].drop(columns="rlnClassNumber")
    write_star(b, proj.cycle_input(0))
    run_cycle(proj, 0, stop="classify")
    m = read_manifest(proj.cycle_dir(0) / "05_classify")
    assert m["info"]["n_classes"] == 1 and "no column rlnClassNumber" in m["warnings"][0]
