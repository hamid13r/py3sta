import numpy as np
import pandas as pd
import pytest

from py3sta.io.particles import ParticleSet, tomo_key
from py3sta.io.star import read_star, write_star
from py3sta.utils.add_warp_xml import WARP_XML_COLUMN, attach_warp_xml

XML = '<?xml version="1.0" encoding="utf-16"?>\n<TiltSeries AreAnglesInverted="True">\n<Angles>-40\n0\n50</Angles>\n</TiltSeries>'


def test_tomo_key():
    for n in ["TS_01.mrc_6.65Apx.mrc", "TS_01.mrc.tomostar", "TS_01.mrc.xml", "TS_01.tomostar.xml",
              "TS_01.xml", "/a/b/TS_01.mrc"]:
        assert tomo_key(n) == "TS_01"


def _setup(tmp_path):
    (tmp_path / "xml").mkdir()
    for n in ("TS_01", "TS_02"):
        (tmp_path / "xml" / f"{n}.mrc.xml").write_bytes(XML.encode("utf-16"))
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnImagePixelSize": [6.65]})
    parts = pd.DataFrame({"rlnCoordinateX": [1.0, 2, 3], "rlnCoordinateY": [1.0, 2, 3], "rlnCoordinateZ": [1.0, 2, 3],
                          "rlnMicrographName": ["TS_01.mrc.tomostar", "TS_02.mrc.tomostar", "TS_01.mrc.tomostar"]})
    write_star({"optics": optics, "particles": parts}, tmp_path / "in.star")
    return tmp_path / "in.star"


def test_attach_absolute_and_relative(tmp_path):
    star = _setup(tmp_path)
    (tmp_path / "out").mkdir()
    m = attach_warp_xml(star, [tmp_path / "xml"], tmp_path / "out" / "a.star")
    assert set(m) == {"TS_01", "TS_02"}
    b = read_star(tmp_path / "out" / "a.star")
    assert set(b) == {"optics", "particles"}                       # optics block preserved
    col = b["particles"][WARP_XML_COLUMN]
    assert col[0] == col[2] != col[1] and col[0].startswith("/")
    attach_warp_xml(star, [str(tmp_path / "xml" / "*.xml")], tmp_path / "out" / "r.star", relative=True)
    r = read_star(tmp_path / "out" / "r.star")["particles"][WARP_XML_COLUMN]
    assert r[0] == "../xml/TS_01.mrc.xml"
    # the average command resolves it relative to the STAR file
    from py3sta.average import per_tomo_files
    ps = ParticleSet.from_star(tmp_path / "out" / "r.star")
    files = per_tomo_files(ps, tmp_path / "out" / "r.star", WARP_XML_COLUMN)
    assert files["TS_02"].name == "TS_02.mrc.xml"


def test_attach_missing(tmp_path):
    star = _setup(tmp_path)
    (tmp_path / "xml" / "TS_02.mrc.xml").unlink()
    with pytest.raises(FileNotFoundError, match="TS_02"):
        attach_warp_xml(star, [tmp_path / "xml"], tmp_path / "o.star")
    attach_warp_xml(star, [tmp_path / "xml"], tmp_path / "o.star", allow_missing=True)
    col = read_star(tmp_path / "o.star")["particles"][WARP_XML_COLUMN]
    assert col[1] in ("", '""') or pd.isna(col[1])
