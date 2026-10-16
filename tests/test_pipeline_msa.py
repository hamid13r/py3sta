import numpy as np
import pytest

from py3sta.average import box_grid
from py3sta.geometry import euler_to_matrix
from py3sta.io.mrc import read_mrc, write_mrc
from py3sta.io.star import read_star, write_star
from py3sta.pipeline import ConfigError, load_project
from py3sta.pipeline.manifest import read_manifest
from py3sta.pipeline.runner import run_cycle, step_state
from py3sta.wedge import TiltInfo, wedge_weights

from test_align import APIX, chiral_object
from test_msa import EXTRA
from test_pipeline_run import _edit_yaml

TOMO = 96
XML = ('<?xml version="1.0" encoding="utf-16"?>\n<TiltSeries AreAnglesInverted="False">\n<Angles>'
       + "\n".join(str(a) for a in range(-50, 51, 5)) + "</Angles>\n</TiltSeries>")


@pytest.fixture
def two_state_proj(tmp_path):
    """4 tomograms (96^3, 4 Å, tilts ±50°, wedge applied) with 8 particles each: the chiral object
    (state 1) or the object + one blob (state 2), random poses, light noise; box 40. rlnRandomSubset =
    state (the truth, for the tests)."""
    import pandas as pd

    from py3sta.pipeline import init_project

    rng = np.random.default_rng(3)
    d = tmp_path / "data"
    d.mkdir()
    rows = []
    grid = np.array([[x, y, z] for x in (24, 72) for y in (24, 72) for z in (24, 72)], float)
    w = wedge_weights(TOMO, TiltInfo.from_range(-50.0, 50.0), soft_deg=0)
    x = box_grid(TOMO) * APIX
    states = rng.permutation(np.tile([1, 2], 16))
    for k in range(4):
        name = f"TS_{k:02d}"
        vol = np.zeros((TOMO,) * 3, np.float32)
        for i, c in enumerate(grid + rng.uniform(-0.5, 0.5, grid.shape)):
            e = rng.uniform(-180, 180, 3)
            M = euler_to_matrix(*e)
            sh = (c - TOMO // 2) * APIX
            vol += chiral_object(TOMO, APIX, shift_A=sh, M=M)
            s = states[8 * k + i]
            if s == 2:
                p, sig, a = EXTRA
                vol += a * np.exp(-((x - sh - M.T @ np.asarray(p)) ** 2).sum(-1) / (2 * sig ** 2))
            rows.append(dict(rlnCoordinateX=c[0], rlnCoordinateY=c[1], rlnCoordinateZ=c[2], rlnAngleRot=e[0],
                             rlnAngleTilt=e[1], rlnAnglePsi=e[2], rlnMicrographName=f"{name}.mrc",
                             rlnWarpXMLfile=f"{name}.mrc.xml", rlnRandomSubset=int(s)))
        vol = np.fft.irfftn(np.fft.rfftn(vol) * w, s=vol.shape, axes=(0, 1, 2))
        vol += rng.normal(scale=0.05, size=vol.shape)
        write_mrc(d / f"{name}.mrc", vol.astype(np.float32), APIX)
        (d / f"{name}.mrc.xml").write_bytes(XML.encode("utf-16"))
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnImagePixelSize": [APIX], "rlnImageSize": [40]})
    write_star({"optics": optics, "particles": pd.DataFrame(rows)}, d / "in.star")
    p = init_project(tmp_path / "proj", d / "in.star", backend="numpy", jobs=1, margin=3).project
    _edit_yaml(p.cycle_dir(0) / "cycle.yaml",
               lambda c: (c["reference"]["mask"].update(radius=[66.0], soft=12.0),
                          c["select"].update(step=30.0, bin=1, lowpass=10.0, max_shift=24.0),
                          c["align"].update(enable=False),
                          c["msa"].update(bin=2, lowpass=16.0, n_factors=6),
                          c["classify"].update(factors=[1, 2, 3], k=[2, 3], bootstrap=4, n_init=4)))
    return load_project(p.root)


def _accuracy(score, states):
    a = np.mean((score > np.median(score)) == (states == 2))
    return max(a, 1 - a)


def _scores(proj):
    df = read_star(proj.cycle_dir(0) / "04_msa" / "msa.star")["particles"]
    truth = read_star(proj.cycle_input(0))["particles"].set_index("rlnTomoParticleName")["rlnRandomSubset"]
    return df, truth.loc[df["rlnTomoParticleName"]].to_numpy()


def test_msa_step_wmd(two_state_proj):
    proj = two_state_proj
    b = read_star(proj.cycle_input(0))
    b["particles"]["py3staExcluded"] = [1, 1] + [0] * 30
    write_star(b, proj.cycle_input(0))
    run_cycle(proj, 0, stop="msa")
    out = proj.cycle_dir(0) / "04_msa"
    for f in ("msa.star", "scores.npy", "pca.npz", "eigenvolumes.mrcs", "scree.star", "factors.star",
              "scree.png", "eigenvolumes.png", "factors.png"):
        assert (out / f).is_file(), f
    df, states = _scores(proj)
    assert len(df) == 30 and list(df.columns[1:]) == [f"py3staPC{j}" for j in range(1, 7)]
    assert _accuracy(df["py3staPC1"].to_numpy(), states) == 1.0
    np.testing.assert_allclose(np.load(out / "scores.npy"), df.iloc[:, 1:].to_numpy(), atol=1e-4)
    ev, apix = read_mrc(out / "eigenvolumes.mrcs")
    assert ev.shape == (7 * 20, 20, 20) and apix == pytest.approx(2 * APIX)
    info = read_manifest(out)["info"]
    assert 1 not in info["wedge_factors"] and info["wedge_r"][0] < 0.3 and info["n_particles"] == 30 and info["pose_source"] == "align"
    scree = read_star(out / "scree.star")["scree"]
    assert (np.diff(scree["py3staEigenvalue"]) <= 0).all()
    # a classification setting does not touch msa; an msa setting makes it stale
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["classify"].update(column="rlnRandomSubset"))
    assert step_state(load_project(proj.root), 0, "msa").state == "done"
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["msa"].update(n_factors=4))
    assert step_state(load_project(proj.root), 0, "msa").state == "stale"


def test_msa_step_plain_warns_wedge_factors(two_state_proj):
    proj = two_state_proj
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["msa"].update(mode="plain", wedge=False))
    run_cycle(proj, 0, stop="msa")
    m = read_manifest(proj.cycle_dir(0) / "04_msa")
    assert 1 in m["info"]["wedge_factors"] and any("missing-wedge orientation" in w for w in m["warnings"])


def test_msa_disabled_and_validation(two_state_proj):
    proj = two_state_proj
    cy = proj.cycle_dir(0) / "cycle.yaml"
    _edit_yaml(cy, lambda c: c["msa"].update(enable=False))
    with pytest.raises(ConfigError, match="classifies MSA factors"):
        load_project(proj.root).cycle_config(0)
    _edit_yaml(cy, lambda c: c["classify"].update(method="column"))
    run_cycle(proj, 0, stop="msa")
    assert read_manifest(proj.cycle_dir(0) / "04_msa")["info"] == {"enabled": False}
    for upd, msg in (({"bin": 3}, "box 40 is not divisible by 3"), ({"lowpass": 12.0}, "beyond Nyquist"),
                     ({"mode": "pca"}, "msa.mode"), ({"n_factors": 0}, "msa.n_factors")):
        _edit_yaml(cy, lambda c: c["msa"].update(enable=True, **upd))
        with pytest.raises(ConfigError, match=msg):
            load_project(proj.root).cycle_config(0)
        _edit_yaml(cy, lambda c: c.update(msa={"bin": 2, "lowpass": 16.0}))
    _edit_yaml(proj.root / "project.yaml", lambda c: c["wedge"].update(use=False))
    with pytest.raises(ConfigError, match="needs tilt information"):
        load_project(proj.root).cycle_config(0)
    _edit_yaml(cy, lambda c: c["msa"].update(mode="plain", wedge=False))
    load_project(proj.root).cycle_config(0)
