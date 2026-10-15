import numpy as np
import pytest
import yaml

from py3sta.average import box_grid, protein_contrast
from py3sta.io.mrc import read_mrc, write_mrc
from py3sta.io.star import read_star
from py3sta.reference import (bandpass, fit_box, load_volumes, make_mask, masked_cc, masked_normalize,
                              prepare_reference)


def _radial(box, apix=1.0, center=(0, 0, 0)):
    return np.sqrt(((box_grid(box) * apix - np.asarray(center)) ** 2).sum(-1))


def test_sphere_mask_edge():
    m = make_mask(32, 2.0, "sphere", [20.0], soft=8.0)
    r = _radial(32, 2.0)
    assert m[r <= 16].min() == pytest.approx(1) and m[r >= 24].max() == pytest.approx(0)
    assert m[16, 16, 26] == pytest.approx(0.5)                     # x = 10 voxels = 20 Å = radius
    prof = m[16, 16, 16:]
    assert np.all(np.diff(prof) <= 1e-7)                           # monotonic
    hard = make_mask(32, 2.0, "sphere", [20.0], soft=0)
    assert set(np.unique(hard)) <= {0.0, 1.0}


def test_ellipsoid_box_and_center():
    m = make_mask(40, 1.0, "ellipsoid", [15.0, 8.0, 4.0], soft=0)
    c = 20
    assert m[c, c, c + 14] == 1 and m[c, c, c + 16] == 0           # x radius 15
    assert m[c, c + 7, c] == 1 and m[c, c + 9, c] == 0             # y radius 8
    assert m[c + 3, c, c] == 1 and m[c + 5, c, c] == 0             # z radius 4
    b = make_mask(40, 1.0, "box", [10.0, 6.0, 3.0], soft=0)
    assert b.sum() == 21 * 13 * 7
    s = make_mask(40, 1.0, "sphere", [5.0], soft=0, center=(8.0, 0, 0))
    zz, yy, xx = np.nonzero(s)
    assert xx.mean() == pytest.approx(28) and yy.mean() == pytest.approx(20)


def test_mask_file(tmp_path):  # noqa: D103
    write_mrc(tmp_path / "m.mrc", np.full((16,) * 3, 2.0, np.float32), 2.0)
    assert make_mask(16, 2.0, "file", file=tmp_path / "m.mrc").max() == 1.0
    with pytest.raises(ValueError, match="differs from the box"):
        make_mask(20, 2.0, "file", file=tmp_path / "m.mrc")
    with pytest.raises(ValueError, match="voxel size"):
        make_mask(16, 3.0, "file", file=tmp_path / "m.mrc")


def test_bandpass():
    box, apix = 64, 2.0
    x = box_grid(box)[..., 0]
    low = np.cos(2 * np.pi * x / 32)          # 64 Å period
    high = np.cos(2 * np.pi * x / 4)          # 8 Å period
    out = bandpass(low + high, apix, lowpass=20.0)
    np.testing.assert_allclose(out, low, atol=1e-4)
    out = bandpass(low + high + 5.0, apix, highpass=30.0)
    np.testing.assert_allclose(out, high, atol=1e-4)


@pytest.mark.parametrize("n,box", [(64, 72), (63, 72), (72, 64), (71, 64), (16, 16)])
def test_fit_box_keeps_centre(n, box):
    v = np.zeros((n,) * 3, np.float32)
    v[n // 2, n // 2, n // 2 + 1] = 1                                # one voxel right of the centre
    out = fit_box(v, box)
    assert out.shape == (box,) * 3 and out[box // 2, box // 2, box // 2 + 1] == 1 and out.sum() == 1


def test_prepare_reference_and_cc():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(24,) * 3).astype(np.float32)
    mask = make_mask(24, 1.0, "sphere", [8.0], soft=2.0)
    r = prepare_reference(v, 1.0, mask, lowpass=4.0)
    w = mask.astype(np.float64)
    inside = mask > 0
    un = np.zeros_like(r)
    un[inside] = r[inside] / mask[inside]                            # values before applying the mask
    assert abs((w * un).sum() / w.sum()) < 1e-5 and np.sqrt((w * un ** 2).sum() / w.sum()) == pytest.approx(1, abs=1e-4)
    assert np.abs(r[~inside]).max() == 0
    assert masked_cc(prepare_reference(v, 1.0, mask, invert=True), masked_normalize(v, mask), mask) == pytest.approx(-1)


def test_load_volumes_stack(tmp_path):
    vols = np.arange(3 * 8 ** 3, dtype=np.float32).reshape(24, 8, 8)
    write_mrc(tmp_path / "s.mrc", vols, 2.0)
    out = load_volumes(tmp_path / "s.mrc", 10, 2.0)
    assert len(out) == 3 and out[1][5, 5, 5] == vols[8 + 4, 4, 4]
    assert load_volumes(tmp_path / "s.mrc", 10, 1.0)[0].shape == (10,) * 3      # resampled 2 -> 1 Å


def test_protein_contrast():
    r = _radial(32)
    blob = (r < 8).astype(np.float32)
    assert protein_contrast(blob, 10)[0] == "light" and protein_contrast(-blob, 10)[0] == "dark"


# ------------------------------------------------------------------ step
from test_pipeline_run import _edit_yaml, _object, proj  # noqa: E402,F401  (fixture)

from py3sta.pipeline import load_project  # noqa: E402
from py3sta.pipeline.manifest import read_manifest  # noqa: E402
from py3sta.pipeline.runner import run_cycle, step_state  # noqa: E402


def test_reference_global_and_staleness(proj):
    run_cycle(proj, 0)
    f = proj.cycle_dir(0) / "02_reference"
    refs, _ = read_mrc(f / "references.mrcs", mmap=False)
    assert refs.shape == (16, 16, 16)
    tab = read_star(f / "references.star")["references"]
    assert tab["py3staReferenceSource"].tolist() == ["global"] and tab["py3staReferenceCC"][0] > 0.99
    assert step_state(proj, 0, "reference").state == "done"
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["reference"].update(lowpass=50.0))
    assert step_state(proj, 0, "average").state == "done"
    st = step_state(proj, 0, "reference")
    assert st.state == "stale" and st.reasons == ["reference.lowpass 40.0→50.0"]
    run_cycle(proj, 0, start="reference")
    run_cycle(proj, 0, stop="average")
    assert step_state(proj, 0, "reference").reasons == ["upstream average re-run"]


def test_reference_files_without_average(proj, tmp_path):
    obj = _object()
    write_mrc(tmp_path / "ref_a.mrc", fit_box(-obj, 20), 10.0)                 # other box, inverted contrast
    write_mrc(tmp_path / "two.mrcs", np.concatenate([obj, np.roll(obj, 2, 2)]), 10.0)
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["reference"].update(
        mode="file", files=[str(tmp_path / "ref_a.mrc"), str(tmp_path / "two.mrcs")]))
    run_cycle(proj, 0, start="reference")                                       # no average needed
    refs, _ = read_mrc(proj.cycle_dir(0) / "02_reference" / "references.mrcs", mmap=False)
    assert refs.shape == (3 * 16, 16, 16)
    tab = read_star(proj.cycle_dir(0) / "02_reference" / "references.star")["references"]
    assert tab["py3staReferenceFile"].str.endswith("two.mrcs#2").sum() == 1
    run_cycle(proj, 0)                                                          # average, then reference
    m = read_manifest(proj.cycle_dir(0) / "02_reference")
    cc = m["info"]["cc_with_average"]
    assert cc[0] < 0 < cc[1] and any("contrast inverted" in w for w in m["warnings"])
    write_mrc(tmp_path / "ref_a.mrc", fit_box(obj, 20), 10.0)                  # replace the file
    st = step_state(proj, 0, "reference")
    assert st.state == "stale" and "reference file(s) changed: ref_a.mrc" in st.reasons[0]


def test_classes_mode_refused(proj):
    from py3sta.pipeline.config import ConfigError

    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["reference"].update(mode="classes"))
    with pytest.raises(ConfigError, match="not implemented yet"):
        load_project(proj.root).cycle_config(0)


def test_invert_contrast_flips_average(proj):
    run_cycle(proj, 0)
    a, _ = read_mrc(proj.cycle_dir(0) / "01_average" / "average.mrc", mmap=False)
    assert read_manifest(proj.cycle_dir(0) / "01_average")["info"]["protein_contrast"] == "light"
    _edit_yaml(proj.root / "project.yaml", lambda c: c.update(invert_contrast=True))
    p2 = load_project(proj.root)
    assert step_state(p2, 0, "average").reasons == ["invert_contrast False→True"]
    run_cycle(p2, 0, stop="average")
    b, _ = read_mrc(proj.cycle_dir(0) / "01_average" / "average.mrc", mmap=False)
    np.testing.assert_allclose(b, -a, atol=1e-5)
    m = read_manifest(proj.cycle_dir(0) / "01_average")
    assert m["info"]["protein_contrast"] == "dark" and any("protein appears dark" in w for w in m["warnings"])
