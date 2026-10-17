import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from click.testing import CliRunner

from py3sta import chimerax as cx
from py3sta.cli import main
from py3sta.io.mrc import write_mrc


@pytest.fixture
def fake_chimerax():
    """A REST server answering like ChimeraX 1.10 with 'json true' (replies captured from the real program),
    including its error on encoding the result of 'open'. Yields (port, list of received commands)."""
    received, models = [], [0]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            cmd = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)["command"][0]
            received.append(cmd)
            notes, error = [], None
            if cmd.startswith("open"):
                path = cmd.split('"')[1]
                if "missing" in path:
                    error = {"type": "UserError", "message": f"No such file/path: {path}"}
                else:
                    models[0] += 1
                    notes = [f"Opened {path.rsplit('/', 1)[1]} as #{models[0]}, grid size 8,8,8\n"]
                    error = {"type": "AttributeError", "message": "'Volume' object has no attribute 'string'"}
            elif cmd.startswith("volume copy"):
                models[0] += 1
                notes = [f"Opened x copy as #{models[0]}, grid size 8,8,8\n"]
            body = json.dumps({"json values": [], "python values": [], "error": error,
                               "log messages": {"note": notes, "warning": [], "error": [], "bug": []}}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv.server_address[1], received
    srv.shutdown()


def test_open_map_and_stack(fake_chimerax, tmp_path):
    port, received = fake_chimerax
    write_mrc(tmp_path / "one.mrc", np.zeros((8, 8, 8), np.float32), 2.0)
    write_mrc(tmp_path / "three.mrcs", np.zeros((24, 8, 8), np.float32), 2.0)
    assert cx.stack_count(tmp_path / "one.mrc") == (1, 8) and cx.stack_count(tmp_path / "three.mrcs") == (3, 8)
    log = cx.open_maps([tmp_path / "one.mrc", tmp_path / "three.mrcs"], port)
    assert "Opened one.mrc as #1" in log and "three.mrcs: 3 volumes as separate models" in log
    assert received[0] == f'open "{tmp_path / "one.mrc"}" format mrc name "one.mrc"'
    assert received[2:] == ["volume copy #2 subregion 0,0,0,7,7,7", "volume #3 originIndex 4,4,4",
                            'rename #3 "three.mrcs [1]"',
                            "volume copy #2 subregion 0,0,8,7,7,15", "volume #4 originIndex -6,4,4",
                            'rename #4 "three.mrcs [2]"',
                            "volume copy #2 subregion 0,0,16,7,7,23", "volume #5 originIndex -16,4,4",
                            'rename #5 "three.mrcs [3]"', "close #2"]
    with pytest.raises(cx.ChimeraXError, match="No such file"):
        cx.run('open "/missing.mrc" format mrc', port)
    r = CliRunner().invoke(main, ["chimerax", str(tmp_path / "one.mrc"), "--port", str(port)])
    assert r.exit_code == 0 and "Opened one.mrc" in r.output


def test_not_running(tmp_path):
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]                      # bound, not listening: connection refused
        with pytest.raises(cx.ChimeraXError, match=f"remotecontrol rest start port {port}"):
            cx.run("version", port)


def test_dashboard_button(fake_chimerax, tmp_path):
    pytest.importorskip("dash")
    from py3sta.dashboard.app import chimerax_open

    port, received = fake_chimerax
    from py3sta.pipeline import init_project
    from test_pipeline_run import _dataset

    p = init_project(tmp_path / "proj", _dataset(tmp_path), backend="numpy", jobs=1).project
    write_mrc(p.cycle_dir(0) / "x.mrc", np.zeros((8, 8, 8), np.float32), 2.0)
    assert chimerax_open(p.root, 0, "x.mrc", port).startswith("Opened x.mrc as #1")
    assert chimerax_open(p.root, 0, None, port) == "choose a map first"
    with socket_closed() as closed:
        assert chimerax_open(p.root, 0, "x.mrc", closed).startswith("ChimeraX: ChimeraX not reachable")


class socket_closed:
    def __enter__(self):
        import socket

        self.s = socket.socket()
        self.s.bind(("127.0.0.1", 0))
        return self.s.getsockname()[1]

    def __exit__(self, *a):
        self.s.close()
