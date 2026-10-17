import json
import socket
import struct
import zlib

import numpy as np
import pytest
from click.testing import CliRunner

from py3sta.cli import main
from py3sta.dashboard import data
from py3sta.pipeline import load_project
from py3sta.pipeline.manifest import LOG_FILE, read_manifest, run_step
from py3sta.pipeline.runner import run_cycle

from test_pipeline_run import _edit_yaml, proj  # noqa: F401  (fixture)


def _decode_png(b: bytes) -> np.ndarray:
    """Minimal reader for the 8-bit grayscale PNGs png_bytes writes (filter 0 rows)."""
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w, h = 8, b"", None, None
    while pos < len(b):
        n, tag = struct.unpack(">I4s", b[pos:pos + 8])
        body = b[pos + 8:pos + 8 + n]
        assert zlib.crc32(tag + body) & 0xFFFFFFFF == struct.unpack(">I", b[pos + 8 + n:pos + 12 + n])[0]
        if tag == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
            assert body[8:10] == b"\x08\x00"
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, w + 1)
    assert (raw[:, 0] == 0).all()
    return raw[:, 1:]


def test_png_and_views():
    img = (np.arange(12 * 7) % 256).astype(np.uint8).reshape(7, 12)
    np.testing.assert_array_equal(_decode_png(data.png_bytes(img)), img)
    vol = np.zeros((6, 8, 10), np.float32)
    vol[1, 2, 3] = 5
    assert data.view(vol, 0, 1)[2, 3] == 5 and data.view(vol, 0).sum() == 0          # slice (default centre)
    assert data.view(vol, 1, project=True).shape == (6, 10) and data.view(vol, 2, project=True)[1, 2] == 5
    assert data.to_uint8(np.ones((3, 3))).max() == 0
    u = data.to_uint8(np.linspace(0, 1, 100).reshape(10, 10), 0, 100)
    assert u.min() == 0 and u.max() == 255


def test_step_log_file(tmp_path):
    lines = []

    def ok(ctx):
        ctx.log("hello")
        ctx.warn("careful")

    run_step(tmp_path, tmp_path / "s", "s", {}, ok, log=lines.append)
    text = (tmp_path / "s" / LOG_FILE).read_text()
    assert lines == ["hello", "warning: careful"] and text.endswith("hello\n" + text.split("\n")[1] + "\n")
    assert "warning: careful" in text and text[2] == ":"                       # HH:MM:SS prefix

    def bad(ctx):
        ctx.log("before")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_step(tmp_path, tmp_path / "s", "s", {}, bad)
    text = (tmp_path / "s" / LOG_FILE).read_text()
    assert "hello" not in text and "before" in text and "RuntimeError: boom" in text and "Traceback" in text


def test_overview_and_logs(proj):
    pytest.importorskip("dash")
    from py3sta.dashboard.app import log_view, overview_page

    ov = data.overview(proj)
    assert [c.state for c in ov.cycles[0].steps] == ["missing"] * len(ov.cycles[0].steps)
    assert ov.cycles[0].fsc143 is None and data.resolution_history(ov) == []
    assert "not run yet" in str(log_view(proj.root, 0, "average"))
    run_cycle(proj, 0, stop="reference")
    ov = data.overview(load_project(proj.root))
    row = ov.cycles[0]
    states = {c.name: c.state for c in row.steps}
    assert states["average"] == "done" and states["reference"] == "done" and states["final"] == "missing"
    assert row.n_particles == 40 and row.n_used == 40 and row.fsc143 > 0
    cell = next(c for c in row.steps if c.name == "average")
    assert cell.n_warnings == 1 and cell.seconds is not None                     # low-CC tomogram TS_04
    assert data.resolution_history(ov) == [(0, row.fsc143, row.fsc143_last, 40)]
    r, f = data.fsc_curve(proj, 0)
    assert len(r) == len(f) and f[0] > 0.9
    thumb = data.thumbnail(data.cycle_average(proj, 0))
    import base64

    img = _decode_png(base64.b64decode(thumb.split(",", 1)[1]))
    box = proj.config.box
    assert img.shape == (box, 3 * box + 4)
    page = str(overview_page(proj.root))
    assert "01_average" in page and "data:image/png;base64" in page and "resolution per cycle" in page
    lg = str(log_view(proj.root, 0, "average"))
    assert "averaging 40 particles" in lg and "TS_04" in lg
    # a stale step is reported with its reason
    _edit_yaml(proj.cycle_dir(0) / "cycle.yaml", lambda c: c["reference"].update(lowpass=25.0))
    cell = next(c for c in data.overview(load_project(proj.root)).cycles[0].steps if c.name == "reference")
    assert cell.state == "stale" and cell.reasons == ["reference.lowpass 40.0→25.0"]


def test_token_and_cli(proj):
    pytest.importorskip("dash")
    from py3sta.dashboard.app import COOKIE, create_app

    c = create_app(proj.root, token="s3cret").server.test_client()
    assert c.get("/").status_code == 403 and c.get("/?token=wrong").status_code == 403
    r = c.get("/?token=s3cret")
    assert r.status_code == 200 and COOKIE in r.headers.get("Set-Cookie", "")
    assert c.get("/_dash-layout").status_code == 200                               # the cookie is enough now
    assert create_app(proj.root).server.test_client().get("/").status_code == 200   # no token
    with socket.socket() as s:                                                       # port in use -> clear error
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        r = CliRunner().invoke(main, ["dashboard", str(proj.root), "--port", str(port)])
    assert r.exit_code != 0 and "in use" in r.output
    r = CliRunner().invoke(main, ["dashboard", str(proj.root / "nope")])
    assert r.exit_code != 0


def test_callbacks_over_http(proj):
    pytest.importorskip("dash")
    from py3sta.dashboard.app import create_app

    run_cycle(proj, 0, stop="average")
    c = create_app(proj.root).server.test_client()

    def call(out, inputs, state=()):
        outs = [dict(zip(("id", "property"), o.split("."))) for o in out.split("...")]
        body = {"output": f"..{out}.." if len(outs) > 1 else out, "outputs": outs if len(outs) > 1 else outs[0],
                "changedPropIds": [],
                "state": [{"id": i, "property": p, "value": v} for i, p, v in state], "inputs": [{"id": i, "property": p, "value": v} for i, p, v in inputs]}
        r = c.post("/_dash-update-component", json=body)
        assert r.status_code == 200, r.data[:500]
        return json.dumps(r.get_json())

    assert "01_average" in call("overview.children", [("tick", "n_intervals", 1), ("tabs", "value", "cycles")])
    assert "averaging 40 particles" in call("log.children", [("log-cycle", "value", 0), ("log-step", "value", "average"),
                                                           ("tick", "n_intervals", 1)])
    assert "cycle-000" in call("log-cycle.options...map-cycle.options...plots-cycle.options", [("tick", "n_intervals", 1)])
    assert "01_average/average.mrc" in call("map-file.options...map-file.value", [("map-cycle", "value", 0)],
                                            [("map-file", "value", None)])
    out = call("map-ortho.figure...map-gallery.figure...map-gallery.style",
               [("map-cycle", "value", 0), ("map-file", "value", "01_average/average.mrc"), ("map-k", "value", 1),
                ("map-mode", "value", "projections"), ("map-lowpass", "value", 30), ("map-mask", "value", []),
                ("map-z", "value", 50), ("map-y", "value", 50), ("map-x", "value", 50)])
    assert "heatmap" in out and "XY projection" in out
    assert "half-map FSC" in call("plots.children", [("plots-cycle", "value", 0), ("tabs", "value", "plots")])


def test_maps_and_plots(proj):
    pytest.importorskip("dash")
    from py3sta.dashboard.app import create_app, gallery_figure, map_figure, plots_page
    from py3sta.io.mrc import write_mrc

    run_cycle(proj, 0, stop="reference")
    maps = data.cycle_maps(proj, 0)
    assert {"01_average/average.mrc", "02_reference/mask.mrc", "02_reference/references.mrcs"} <= set(maps)
    box, apix = proj.config.box, proj.config.apix
    stack = np.random.default_rng(0).normal(size=(3 * box, box, box)).astype(np.float32)
    stack[box + 3, 5, 7] = 100                                                   # volume 2, z=3, y=5, x=7
    write_mrc(proj.cycle_dir(0) / "02_reference" / "test.mrcs", stack, apix)
    vols = data.load_volumes(proj.cycle_dir(0) / "02_reference" / "test.mrcs")
    assert vols.shape == (3, box, box, box) and vols[1, 3, 5, 7] == 100
    f = map_figure(proj.root, 0, "02_reference/test.mrcs", 1, False, None, False,
                   (100 * 3 / (box - 1), 100 * 5 / (box - 1), 100 * 7 / (box - 1)))
    xy, xz, yz = (np.asarray(t.z) for t in f.data)
    assert xy[5, 7] == 100 and xz[3, 7] == 100 and yz[3, 5] == 100            # slices through the voxel
    assert "[2/3]" in f.layout.title.text
    f = map_figure(proj.root, 0, "01_average/average.mrc", 0, True, 30.0, True)
    assert len(f.data) == 6 and "low-pass 30 Å" in f.layout.title.text      # 3 projections + 3 mask contours
    assert len(gallery_figure(proj.root, 0, "02_reference/test.mrcs").data) == 3
    assert len(gallery_figure(proj.root, 0, "01_average/average.mrc").data) == 0    # one map: no gallery
    lp = data.filtered(vols[1], apix, 4 * apix)
    assert lp.std() < vols[1].std() and np.array_equal(data.filtered(vols[1], apix, apix), vols[1])   # beyond Nyquist: unchanged
    page = str(plots_page(proj.root, 0))
    assert "half-map FSC" in page
    c = create_app(proj.root).server.test_client()
    (proj.cycle_dir(0) / "01_average" / "x.png").write_bytes(data.png_bytes(np.zeros((2, 2), np.uint8)))
    assert "cycle-000/01_average/x.png" in str(plots_page(proj.root, 0))
    assert c.get("/files/cycle-000/01_average/x.png").status_code == 200
    assert c.get("/files/cycle-000/01_average/average.mrc").status_code == 404
    assert c.get("/files/cycle-000/%2E%2E/%2E%2E/x.png").status_code == 404
    with pytest.raises(ValueError, match="outside the project"):
        data.project_file(proj, "../x.png")
