import numpy as np
import pandas as pd
import pytest

from py3sta.align import mirror
from py3sta.geometry import euler_to_matrix
from py3sta.io.mrc import open_volume_stack, write_mrc
from py3sta.io.particles import ParticleSet
from py3sta.io.star import read_star, write_star
from py3sta.pipeline import ConfigError, load_project
from py3sta.pipeline.manifest import read_manifest
from py3sta.pipeline.runner import run_cycle, step_state
from py3sta.poses import apply_delta, invert

from test_align import APIX, chiral_object
from test_pipeline_run import _edit_yaml
from test_pipeline_select import _angles, chiral_proj  # noqa: F401  (fixture)

LEVELS = [[12.0, 4.0, 12.0, 4.0], [3.0, 1.0, 3.0, 1.0], [1.0, 0.5, 1.0, 0.5]]


def _setup(proj, refs=None, exclude_first=0):
    """Poses perturbed per particle by a known delta; reference(s) = the true object (file mode)."""
    pytest.importorskip("torch")
    cin = proj.cycle_input(0)
    true = ParticleSet.from_star(cin)
    n = len(true)
    rng = np.random.default_rng(11)
    R = euler_to_matrix(*rng.uniform(-5, 5, (3, n)))
    t = rng.uniform(-0.8, 0.8, (n, 3))                                          # margin 3: shift limit 1 voxel
    wrong = apply_delta(true, *invert(R, t), APIX)
    b = read_star(cin)
    for c in ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]:
        b["particles"][c] = wrong.df[c].to_numpy()
    if exclude_first:
        b["particles"]["py3staExcluded"] = [1] * exclude_first + [0] * (n - exclude_first)
    write_star(b, cin)
    refs = [chiral_object()] if refs is None else refs
    data = open_volume_stack(proj.root / "refs.mrcs", len(refs), refs[0].shape[0], np.float32, APIX, mode="w+")
    data[:] = np.stack(refs)
    data.flush()
    _edit_yaml(proj.root / "project.yaml", lambda c: c["compute"].update(backend="torch", gpu="cpu"))
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml",
               lambda c: (c["reference"].update(mode="file", files=["refs.mrcs"], lowpass=10.0),
                          c["align"].update(enable=True, levels=LEVELS, lowpass=10.0)))
    return load_project(proj.root), true, wrong


def test_align_step_recovers_poses(chiral_proj):
    proj, true, wrong = _setup(chiral_proj, exclude_first=2)
    run_cycle(proj, 0, stop="align")
    out = proj.cycle_dir(0) / "03_align"
    al = ParticleSet.from_star(out / "aligned.star")
    df = al.df
    act = df["py3staExcluded"].to_numpy() == 0
    err = _angles(al, true)[act]
    shift = np.abs(al.centers_px - true.centers_px).max(1)[act] * true.ts_apix / APIX
    # one particle ends 2.2° off with a HIGHER CC than at its true pose (0.9640 vs 0.9624): the score's limit
    # on this smooth, noisy object, not a search miss
    assert np.median(err) < 1.0 and err.max() < 2.5, err
    assert shift.max() < 0.3, shift
    np.testing.assert_allclose(_angles(al, wrong)[~act], 0, atol=1e-6)               # excluded: untouched
    assert (df.loc[act, "rlnBestReference"] == 1).all() and (df.loc[~act, "rlnBestReference"] == 0).all()
    assert "rlnClassNumber" not in df                                               # left to classification
    assert (df.loc[act, "py3staRefCC"] > 0.8).all() and df.loc[act, "py3staAlignEdge"].sum() == 0
    csv = pd.read_csv(out / "align.csv")
    assert len(csv) == 14 and {"cc_ref1", "angle_deg", "shift_x_A"} <= set(csv.columns)
    info = read_manifest(out)["info"]
    assert info["n_aligned"] == 14 and info["n_at_edge"] == 0 and info["pose_source"] == "input"


def test_downstream_uses_aligned_poses(chiral_proj):
    proj, true, _ = _setup(chiral_proj)
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["select"].update(align=[]))
    run_cycle(proj, 0)
    assert read_manifest(proj.cycle_dir(0) / "05_classify")["info"]["pose_source"] == "align"
    fin = ParticleSet.from_star(proj.cycle_dir(0) / "08_final" / "cycle-000_final.star")
    assert np.median(_angles(fin, true)) < 1.0
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["align"].update(max_shift=4.0))
    st = step_state(load_project(proj.root), 0, "classify")
    assert st.state == "stale" and st.reasons == ["upstream align stale"]


def test_multi_reference_best_reference(chiral_proj):
    proj, true, _ = _setup(chiral_proj, refs=[mirror(chiral_object()), chiral_object()])
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["align"].update(bin=2, lowpass=16.0, levels=LEVELS[:2]))
    run_cycle(proj, 0, stop="align")
    df = read_star(proj.cycle_dir(0) / "03_align" / "aligned.star")["particles"]
    assert (df["rlnBestReference"] == 2).all()
    assert read_manifest(proj.cycle_dir(0) / "03_align")["info"]["best_reference_counts"] == {"1": 0, "2": 16}


def test_align_config_validation(chiral_proj):
    cy = chiral_proj.cycle_dir(0) / "cycle.yaml"
    for upd, msg in (({"max_shift": 6.0}, "exceeds the stack drift allowance"),        # margin 3 at 4 Å: 4 Å
                     ({"bin": 3}, "box 40 is not divisible by 3"),
                     ({"bin": 2, "lowpass": 12.0}, "beyond Nyquist of the binned search"),
                     ({"levels": [[10.0, 0.0, 5.0, 5.0]]}, "align.levels\\[0\\]"),
                     ({"wedge": True}, "needs tilt information")):
        _edit_yaml(cy, lambda c: c["align"].update(**{"enable": True, "wedge": False, **upd}))
        with pytest.raises(ConfigError, match=msg):
            load_project(chiral_proj.root).cycle_config(0)
        _edit_yaml(cy, lambda c: c.update(align={"enable": False}))
    p = load_project(chiral_proj.root)
    assert p.config.align_shift_vox(None) == [1, 1, 1] and p.config.align_shift_vox(3.9) == [0, 0, 0]
    assert p.config.align_shift_vox([4.0, 4.0, 0.0]) == [1, 1, 0]
    for upd, msg in (({"max_shift": [4.0, 4.0]}, "one value or \\[x, y, z\\]"), ({"min_gain": -0.1}, "min_gain")):
        _edit_yaml(cy, lambda c: c["align"].update(**{"enable": True, "wedge": False, **upd}))
        with pytest.raises(ConfigError, match=msg):
            load_project(chiral_proj.root).cycle_config(0)


def test_per_axis_shift_and_min_gain(chiral_proj):
    """z shift limit 0: no particle moves along tomogram z; a huge min_gain keeps every pose."""
    proj, true, wrong = _setup(chiral_proj)
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml",
               lambda c: c["align"].update(bin=2, lowpass=16.0, levels=LEVELS[:2], max_shift=[4.0, 4.0, 0.0]))
    run_cycle(proj, 0, stop="align")
    al = ParticleSet.from_star(proj.cycle_dir(0) / "03_align" / "aligned.star")
    d = (al.centers_px - wrong.centers_px) * al.ts_apix / APIX
    np.testing.assert_allclose(d[:, 2], 0, atol=1e-6)
    assert np.abs(d[:, :2]).max() > 0.1
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["align"].update(min_gain=5.0))
    run_cycle(load_project(proj.root), 0, start="align", stop="align")
    al = ParticleSet.from_star(proj.cycle_dir(0) / "03_align" / "aligned.star")
    np.testing.assert_allclose(_angles(al, wrong), 0, atol=1e-4)
    assert read_manifest(proj.cycle_dir(0) / "03_align")["info"]["n_unchanged"] == 16
