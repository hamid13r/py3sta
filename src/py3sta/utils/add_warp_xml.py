"""Attach Warp tilt-series XML files to a particle STAR file.

Adds a per-particle column ``rlnWarpXMLfile`` pointing at the Warp/WarpTools
``<tilt series>.xml`` of the particle's tomogram, so later steps (wedge model,
dose weighting, and later CTF) read tilt information straight from the STAR
file. Particles are matched to XML files by tomogram name (``tomo_key``):
``TS_01.mrc.xml`` / ``TS_01.xml`` / ``TS_01.tomostar.xml`` <-> rlnTomoName
``TS_01`` (or rlnMicrographName ``TS_01.mrc_6.65Apx.mrc`` / ``TS_01.mrc.tomostar``).

``rlnWarpXMLfile`` is not a RELION label; RELION ignores unknown labels with a
warning, and py3sta / starfile keep it.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

from ..io.particles import tomo_key, tomo_names_of
from ..io.star import read_star, write_star

WARP_XML_COLUMN = "rlnWarpXMLfile"


def find_xml_files(sources) -> list[Path]:
    """Expand files, directories (non-recursive) and glob patterns into .xml paths."""
    out = []
    for s in sources:
        p = Path(s)
        if p.is_dir():
            out += sorted(p.glob("*.xml"))
        elif p.is_file():
            out.append(p)
        else:
            out += [Path(g) for g in sorted(glob.glob(str(s)))]
    return [p for p in dict.fromkeys(out) if p.suffix == ".xml"]


def particles_block(blocks: dict) -> str:
    if "particles" in blocks:
        return "particles"
    for name, df in blocks.items():
        if "rlnCoordinateX" in df or "rlnCenteredCoordinateXAngst" in df:
            return name
    raise ValueError("no particle block (with rlnCoordinateX) in STAR file")


def attach_warp_xml(star_in, sources, star_out, relative: bool = False, allow_missing: bool = False) -> dict:
    """Write ``star_out`` = ``star_in`` + rlnWarpXMLfile column. Returns {tomo: xml path}.

    relative=True writes paths relative to ``star_out``'s directory (RELION
    project style); otherwise absolute paths. Unmatched tomograms raise unless
    ``allow_missing`` (then their particles get an empty value).
    """
    xmls = find_xml_files(sources)
    if not xmls:
        raise FileNotFoundError(f"no .xml files found in {list(sources)}")
    by_key: dict[str, Path] = {}
    for x in xmls:
        k = tomo_key(x)
        if k in by_key and by_key[k].resolve() != x.resolve():
            raise ValueError(f"two XML files map to tomogram {k!r}: {by_key[k]} and {x}")
        by_key[k] = x

    blocks = read_star(star_in)
    name = particles_block(blocks)
    df = blocks[name].copy()
    tomos = tomo_names_of(df)
    wanted = list(dict.fromkeys(tomos))
    missing = [t for t in wanted if t not in by_key]
    if missing and not allow_missing:
        raise FileNotFoundError(f"no XML for {len(missing)} tomogram(s): {', '.join(missing[:5])}"
                                f"; XML files found for: {', '.join(sorted(by_key)[:5])}")
    out_dir = Path(star_out).resolve().parent

    def fmt(p: Path) -> str:
        p = p.resolve()
        return os.path.relpath(p, out_dir) if relative else str(p)

    mapping = {t: by_key[t] for t in wanted if t in by_key}
    df[WARP_XML_COLUMN] = tomos.map(lambda t: fmt(mapping[t]) if t in mapping else "")
    blocks[name] = df
    write_star(blocks, star_out)
    return mapping


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="python -m py3sta.utils.add_warp_xml",
                                 description="Add rlnWarpXMLfile (Warp tilt-series XML per tomogram) to a particle STAR file.")
    ap.add_argument("star", help="input particle STAR file")
    ap.add_argument("xml", nargs="+", help="XML files, directories or glob patterns")
    ap.add_argument("-o", "--output", help="output STAR (default: <input>_xml.star)")
    ap.add_argument("--relative", action="store_true", help="write paths relative to the output STAR file")
    ap.add_argument("--allow-missing", action="store_true", help="leave tomograms without an XML empty instead of failing")
    a = ap.parse_args(argv)
    out = a.output or str(Path(a.star).with_suffix("")) + "_xml.star"
    m = attach_warp_xml(a.star, a.xml, out, a.relative, a.allow_missing)
    for t, p in m.items():
        print(f"{t}: {p}")
    print(f"{len(m)} tomogram(s) matched -> {out}")


if __name__ == "__main__":
    main()
