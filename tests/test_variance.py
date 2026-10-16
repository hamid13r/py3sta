import json

import numpy as np
import pytest
from click.testing import CliRunner

from py3sta.average import box_grid, extraction_sphere
from py3sta.cli import main
from py3sta.io.mrc import read_mrc, write_mrc
from py3sta.msa import MSASettings, ParticleVolumes, binned_mask
from py3sta.reference import make_mask
from py3sta.variance import region_stats, variance_map

from test_align import APIX, chiral_object
from test_msa import EXTRA, MASK, TILT, two_states
from test_pipeline_msa import two_state_proj  # noqa: F401  (fixture)

S = dict(bin=2, lowpass=16.0)
TILTS = {"TS_01": TILT}


def test_wmd_variance_finds_the_variable_blob():
    stack, ps, _ = two_states()
    res = variance_map(ps, stack, MASK, APIX, MSASettings("wmd", **S), TILTS)
    v = res.variance
    g = box_grid(res.box) * res.apix
    near = np.linalg.norm(g - np.asarray(EXTRA[0]), axis=-1) < 8
    assert near[np.unravel_index(v.argmax(), v.shape)]
    assert v[near].mean() > 4 * np.median(v[extraction_sphere(res.box) > 0])
    assert (v[extraction_sphere(res.box) == 0] == 0).all() and res.n == 80


def test_one_state_wmd_flat_plain_shows_wedge():
    """No heterogeneity: wmd variance on the object ~ solvent; plain is inflated there by the wedge."""
    stack, ps, _ = two_states(noise=0.1, extra=None)
    obj = chiral_object(20, 2 * APIX)
    on = obj > 0.3 * obj.max()
    off = (extraction_sphere(20) > 0) & (obj < 0.02 * obj.max())
    ratio = {}
    for name, s in (("wmd", MSASettings("wmd", **S)), ("plain", MSASettings("plain", wedge=False, **S))):
        v = variance_map(ps, stack, MASK, APIX, s, TILTS).variance
        ratio[name] = v[on].mean() / v[off].mean()
    assert ratio["wmd"] < 1.3 and ratio["plain"] > 3, ratio


def test_normalised_inside_norm_mask():
    stack, ps, _ = two_states(n=6)
    norm = make_mask(40, APIX, "sphere", [30.0], soft=0.0, center=[20.0, 0.0, 0.0])
    pv = ParticleVolumes(ps, stack, 40, APIX, MSASettings("plain", wedge=False, **S), norm)
    m = (binned_mask(norm, APIX, 2) > 0.5) & (extraction_sphere(20) > 0)
    for _, vols in pv:
        inside = vols[:, m]
        np.testing.assert_allclose(inside.mean(1), 0, atol=1e-5)
        np.testing.assert_allclose(inside.std(1), 1, atol=1e-4)


def test_torch_matches_numpy_and_stats():
    torch = pytest.importorskip("torch")
    stack, ps, _ = two_states(n=10)
    for s in (MSASettings("wmd", **S), MSASettings("plain", wedge=True, **S)):
        a = variance_map(ps, stack, MASK, APIX, s, TILTS)
        b = variance_map(ps, stack, MASK, APIX, s, TILTS, device=torch.device("cpu"))
        np.testing.assert_allclose(b.variance, a.variance, rtol=1e-3, atol=1e-4 * a.variance.max())
    st = region_stats(a, MASK, APIX, 2, MASK)                 # same mask twice: outside = solvent baseline
    assert st["variance_mask"]["voxels"] == st["norm_mask"]["voxels"] and st["baseline"] == "outside"
    assert "median_over_baseline" in st["variance_mask"]
    with pytest.raises(ValueError, match="at least 2"):
        variance_map(type(ps)(ps.df.iloc[:1], ps.ts_apix), stack, MASK, APIX, MSASettings("plain", wedge=False, **S))


def test_cli(two_state_proj, tmp_path):
    proj = two_state_proj
    from py3sta.pipeline.runner import run_cycle

    run_cycle(proj, 0, stop="align")
    m = make_mask(40, APIX, "sphere", [14.0], soft=4.0, center=list(EXTRA[0]))
    write_mrc(tmp_path / "blob.mrc", m, APIX)
    r = CliRunner().invoke(main, ["variance", str(proj.root), "0", "--norm-mask", "reference",
                                  "--mask", str(tmp_path / "blob.mrc")])
    assert r.exit_code == 0, r.output
    assert "variance mask" in r.output and "x baseline" in r.output
    out = proj.cycle_dir(0) / "variance" / "mask_in_blob_wmd"
    for f in ("variance.mrc", "std.mrc", "average.mrc", "variance_masked.mrc", "stats.json", "variance.png",
              "variance_slices.png"):
        assert (out / f).is_file(), f
    info = json.loads((out / "stats.json").read_text())
    assert info["n_particles"] == 32 and info["mode"] == "wmd" and info["bin"] == 2
    assert info["stats"]["variance_mask"]["median_over_baseline"] > 1.5
    assert read_mrc(out / "variance.mrc")[0].shape == (20, 20, 20)
    r = CliRunner().invoke(main, ["variance", str(proj.root), "0", "--norm-mask", str(tmp_path / "blob.mrc"),
                                  "--mask", str(tmp_path / "blob.mrc"), "--mode", "plain", "--no-wedge",
                                  "--bin", "1", "--lowpass", "10", "-o", str(tmp_path / "v")])
    assert r.exit_code == 0, r.output
    assert json.loads((tmp_path / "v" / "stats.json").read_text())["box"] == 40
    write_mrc(tmp_path / "small.mrc", np.zeros((20,) * 3, np.float32), APIX)
    r = CliRunner().invoke(main, ["variance", str(proj.root), "0", "--norm-mask", str(tmp_path / "small.mrc")])
    assert r.exit_code != 0 and "not the project box" in r.output
