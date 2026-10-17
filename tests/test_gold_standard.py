import numpy as np
import pytest

from py3sta.geometry import euler_to_matrix
from py3sta.io.mrc import read_mrc, write_mrc
from py3sta.io.particles import ParticleSet
from py3sta.io.star import read_star, write_star
from py3sta.pipeline import ConfigError, load_project
from py3sta.pipeline.manifest import read_manifest
from py3sta.pipeline.runner import run_cycle, step_state
from py3sta.poses import apply_delta, invert
from py3sta.reference import prepare_reference

from test_align import APIX, chiral_object
from test_pipeline_align import LEVELS
from test_pipeline_run import _edit_yaml, _final
from test_pipeline_select import _angles, _common_rotation_spread, chiral_proj  # noqa: F401  (fixture)

D = euler_to_matrix(6.0, 5.0, -4.0)                         # common pose error of half set 2 (~9°)
POSES = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]


def _write_poses(proj, ps, extra=None):
    cin = proj.cycle_input(0)
    b = read_star(cin)
    for c in POSES:
        b["particles"][c] = ps.df[c].to_numpy()
    for c, v in (extra or {}).items():
        b["particles"][c] = v
    write_star(b, cin)


def _half2_rotated(proj):
    """Half set 2 = 4 of the 16 particles, all with the same pose error D: the half-2 average is the object
    in another orientation, the full average is dominated by half 1."""
    pytest.importorskip("torch")
    true = ParticleSet.from_star(proj.cycle_input(0))
    half = np.where(np.arange(len(true)) % 4 == 3, 2, 1)
    wrong = apply_delta(true, *invert(D, np.zeros(3)), APIX, rows=half == 2)
    _write_poses(proj, wrong, {"rlnRandomSubset": half})
    _edit_yaml(proj.root / "project.yaml", lambda c: c["compute"].update(backend="torch", gpu="cpu"))
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml",
               lambda c: (c["reference"].update(lowpass=10.0),
                          c["align"].update(enable=True, levels=LEVELS, lowpass=10.0)))
    return load_project(proj.root), half, wrong


def test_gold_standard_align_uses_own_half(chiral_proj):
    proj, half, wrong = _half2_rotated(chiral_proj)
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c.update(gold_standard=True))
    proj = load_project(proj.root)
    run_cycle(proj, 0, stop="align")
    ref = proj.cycle_dir(0) / "02_reference"
    mask = read_mrc(ref / "mask.mrc", mmap=False)[0]
    for k in (1, 2):                                          # half references = the prepared half averages
        h = read_mrc(proj.cycle_dir(0) / "01_average" / f"half{k}.mrc", mmap=False)[0]
        got = read_mrc(ref / f"references_half{k}.mrcs", mmap=False)[0].reshape(-1, *h.shape)[0]
        np.testing.assert_allclose(got, prepare_reference(h, APIX, mask, 10.0, None), atol=1e-5)
    assert read_manifest(ref)["info"]["half_reference_cc"] < 0.995                     # different half maps
    al = ParticleSet.from_star(proj.cycle_dir(0) / "03_align" / "aligned.star")
    moved = _angles(al, wrong)
    gold2 = np.median(moved[half == 2])
    assert gold2 < 2.0 and np.median(moved[half == 1]) < 2.0, moved   # each half fits its own half reference
    assert read_manifest(proj.cycle_dir(0) / "03_align")["info"]["gold_standard"] is True
    # without gold standard half 2 is pulled to the full (half-1 dominated) reference
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c.update(gold_standard=False))
    proj = load_project(proj.root)
    assert step_state(proj, 0, "reference").state == "stale"
    run_cycle(proj, 0, start="reference", stop="align")
    al = ParticleSet.from_star(proj.cycle_dir(0) / "03_align" / "aligned.star")
    plain2 = np.median(_angles(al, wrong)[half == 2])
    assert plain2 > 2.5 and plain2 > 1.5 * gold2, (plain2, gold2)   # blended reference: half 2 pulled towards half 1
    assert not (ref / "references_half1.mrcs").exists()


@pytest.mark.parametrize("target", ["target.mrc", "each"])
def test_gold_standard_select_per_half(chiral_proj, target):
    """Classes by tomogram (crossing the half sets); class 2 perturbed by a known transform; every class is
    averaged and aligned per half set."""
    proj = chiral_proj
    true = ParticleSet.from_star(proj.cycle_input(0))
    tomo2 = true.df["rlnMicrographName"].str.contains("TS_01").to_numpy()
    R2, T2 = euler_to_matrix(20.0, 15.0, -10.0), np.array([1.0, 0.0, -1.0])
    _write_poses(proj, apply_delta(true, *invert(R2, T2), APIX, rows=tomo2), {"myClass": np.where(tomo2, 2, 1)})
    write_mrc(proj.root / "target.mrc", chiral_object(), APIX)
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml",
               lambda c: (c.update(gold_standard=True), c["classify"].update(column="myClass"),
                          c["select"].update(target=target, anchor=1 if target == "each" else None)))
    run_cycle(load_project(proj.root), 0)
    sel = proj.cycle_dir(0) / "07_select"
    tr = read_star(sel / "transforms.star")["transforms"]
    assert tr[["py3staClass", "py3staHalf"]].values.tolist() == [[1, 1], [2, 1], [1, 2], [2, 2]]
    assert (sel / "classes_aligned_half1.mrcs").is_file() and not (sel / "classes_aligned.mrcs").exists()
    fin = ParticleSet.from_star(_final(proj))
    if target == "each":                                       # anchor class 1 keeps its (true) poses
        assert _common_rotation_spread(fin, true) < 1.0
        assert tr.loc[tr["py3staClass"] == 1, "py3staClassCC"].isna().all()
    assert _angles(fin, true).max() < 2.0                   # 4-particle half-class averages: ~1.4° noise
    df = read_star(_final(proj))["particles"]
    assert df["py3staClassCC"].notna().sum() == (16 if target != "each" else 8)
    info = read_manifest(sel)["info"]
    assert info["gold_standard"] is True and set(info["class_cc"]) == {"1/1", "2/1", "1/2", "2/2"}


def test_gold_standard_validation(chiral_proj):
    proj = chiral_proj
    cy = proj.cycle_dir(0) / "cycle.yaml"
    _edit_yaml(cy, lambda c: (c.update(gold_standard=True), c["average"].update(halves=False)))
    with pytest.raises(ConfigError, match="gold_standard: needs average.halves"):
        load_project(proj.root).cycle_config(0)
    _edit_yaml(cy, lambda c: (c["average"].update(halves=True), c["align"].update(enable=True, levels=LEVELS)))
    b = read_star(proj.cycle_input(0))
    b["particles"]["rlnRandomSubset"] = 1
    write_star(b, proj.cycle_input(0))
    pytest.importorskip("torch")
    _edit_yaml(proj.root / "project.yaml", lambda c: c["compute"].update(backend="torch", gpu="cpu"))
    with pytest.raises(ValueError, match="both half sets 1 and 2"):
        run_cycle(load_project(proj.root), 0, stop="align")
