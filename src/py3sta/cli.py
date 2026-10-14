"""Command line: ``py3sta <command> --help``."""
from __future__ import annotations

import glob
import shutil
from pathlib import Path

import click

from .io.dipole import dipoles_to_particles
from .io.particles import DEFAULT_TOMO_REGEX, ParticleSet


def _expand(patterns):
    files = []
    for p in patterns:
        hits = sorted(glob.glob(p)) or ([p] if Path(p).exists() else [])
        if Path(p).is_dir():
            hits = sorted(str(f) for f in Path(p).iterdir() if f.is_file())
        files += hits
    if not files:
        raise click.UsageError(f"no files matched {patterns}")
    return files


def _warp_hint(apix, star):
    click.echo(f"WarpTools: ts_export_particles --input_star {star} --coords_angpix {apix:g} ...")


def _write_i3_project(ps: ParticleSet, out_dir, tomo_apix, tomo_file_fmt, tlt_template, offset=0.0):
    out = Path(out_dir)
    written = ps.to_trf(out / "trf", tomo_apix, offset=offset)
    (out / "defs").mkdir(parents=True, exist_ok=True)
    (out / "tlt").mkdir(exist_ok=True)
    sets, maps = [], []
    for tomo in written:
        fname = tomo_file_fmt.format(tomo=tomo)
        sets.append(f"{fname} {tomo}")
        maps.append(f"../maps/ {fname} ../tlt/{tomo}.tlt")
        if tlt_template:
            shutil.copyfile(tlt_template, out / "tlt" / f"{tomo}.tlt")
    (out / "defs" / "sets").write_text("\n".join(sets) + "\n")
    (out / "defs" / "maps").write_text("\n".join(maps) + "\n")
    click.echo(f"i3 project files: {len(written)} trf files, defs/sets, defs/maps in {out}")
    if not tlt_template:
        click.echo("  note: no --tlt-template given; put one .tlt per tomogram in tlt/")


@click.group()
def main():
    """py3sta: i3-style subtomogram averaging, RELION 4 / Warp compatible."""


@main.command()
@click.argument("pos", nargs=-1, required=True)
@click.option("--pick-apix", type=float, required=True, help="Pixel size (A) of the tomograms the picks were made on.")
@click.option("--ts-apix", type=float, help="Unbinned tilt-series pixel size (A) for RELION 4 coordinates. Default: --pick-apix.")
@click.option("--random", "random_inplane", is_flag=True, help="Randomise the in-plane angle about the membrane normal.")
@click.option("--seed", type=int, help="Random seed (reproducible --random).")
@click.option("-o", "--output", default="particles.star", show_default=True)
@click.option("--warp", "warp_out", help="Also write a WarpTools-ready STAR here.")
@click.option("--i3-dir", help="Also write an i3 project (trf/, defs/) here, at --i3-apix.")
@click.option("--i3-apix", type=float, help="Pixel size (A) of the tomograms i3 will use. Default: --pick-apix.")
@click.option("--tomo-file-fmt", default="{tomo}.mrc", show_default=True, help="Tomogram file name in maps/ for defs files.")
@click.option("--tlt-template", type=click.Path(exists=True), help="Copy this .tlt for each tomogram (i3 project only).")
@click.option("--strip-regex", default=DEFAULT_TOMO_REGEX, show_default=True, help="Removed from .pos stem to get the tomogram name.")
def pos2star(pos, pick_apix, ts_apix, random_inplane, seed, output, warp_out, i3_dir, i3_apix,
             tomo_file_fmt, tlt_template, strip_regex):
    """Dipole .pos picks (files, globs or a directory) -> RELION 4 particles.star."""
    files = [f for f in _expand(pos) if f.endswith(".pos")]
    ps = dipoles_to_particles(files, pick_apix, ts_apix, random_inplane, seed, strip_regex)
    ps.to_relion4(output)
    click.echo(f"{len(ps)} particles from {len(files)} files -> {output}")
    if warp_out:
        _warp_hint(ps.to_warp(warp_out), warp_out)
    if i3_dir:
        _write_i3_project(ps, i3_dir, i3_apix or pick_apix, tomo_file_fmt, tlt_template)


@main.command()
@click.argument("trf", nargs=-1, required=True)
@click.option("--tomo-apix", type=float, required=True, help="Pixel size (A) of the tomograms i3 used.")
@click.option("--ts-apix", type=float, help="Unbinned tilt-series pixel size (A). Default: --tomo-apix.")
@click.option("-o", "--output", default="particles.star", show_default=True)
@click.option("--warp", "warp_out", help="Also write a WarpTools-ready STAR here.")
@click.option("--strip-regex", default=DEFAULT_TOMO_REGEX, show_default=True)
@click.option("--offset", type=float, default=0.0, help="Voxel offset added before scaling (IMOD binvol data: (bin-1)/2).")
def trf2star(trf, tomo_apix, ts_apix, output, warp_out, strip_regex, offset):
    """i3 .trf / i3datalist output -> RELION 4 particles.star (replaces trf2star.py)."""
    ps = ParticleSet.from_trf(_expand(trf), tomo_apix, ts_apix, strip_regex=strip_regex, offset=offset)
    ps.to_relion4(output)
    click.echo(f"{len(ps)} particles -> {output}")
    if warp_out:
        _warp_hint(ps.to_warp(warp_out), warp_out)


@main.command()
@click.argument("star")
@click.option("--tomo-apix", type=float, required=True, help="Pixel size (A) of the tomograms i3 will use.")
@click.option("--ts-apix", type=float, help="Needed only if the STAR file has no rlnTomoTiltSeriesPixelSize.")
@click.option("-o", "--out-dir", default=".", show_default=True, help="i3 project directory.")
@click.option("--tomo-file-fmt", default="{tomo}.mrc", show_default=True)
@click.option("--tlt-template", type=click.Path(exists=True))
@click.option("--offset", type=float, default=0.0)
def star2trf(star, tomo_apix, ts_apix, out_dir, tomo_file_fmt, tlt_template, offset):
    """RELION 4 / Warp STAR -> i3 project (trf/, defs/sets, defs/maps)."""
    ps = ParticleSet.from_star(star, ts_apix)
    _write_i3_project(ps, out_dir, tomo_apix, tomo_file_fmt, tlt_template, offset)


@main.command()
@click.argument("star")
@click.option("--ts-apix", type=float, help="Needed only if the STAR file has no rlnTomoTiltSeriesPixelSize.")
@click.option("-o", "--output", default="particles_warp.star", show_default=True)
@click.option("--suffix", default=".tomostar", show_default=True)
def star2warp(star, ts_apix, output, suffix):
    """RELION 4 particles.star -> minimal STAR for WarpTools ts_export_particles."""
    ps = ParticleSet.from_star(star, ts_apix)
    _warp_hint(ps.to_warp(output, suffix), output)


@main.command()
@click.argument("star")
@click.argument("tomograms", nargs=-1)
@click.option("--box", type=int, required=True, help="Box size in tomogram voxels.")
@click.option("--tomo-dir", type=click.Path(file_okay=False), help="Directory holding the tomograms named in rlnMicrographName.")
@click.option("--ts-apix", type=float, help="Pixel size of STAR coordinates if not in the file (rlnTomoTiltSeriesPixelSize/rlnImagePixelSize).")
@click.option("-o", "--output", default="average.mrc", show_default=True)
@click.option("--no-rotate", is_flag=True, help="Ignore angles (raw, unaligned sum).")
@click.option("--no-normalize", is_flag=True, help="Don't normalise each subvolume to mean 0 / std 1.")
@click.option("--halves/--no-halves", default=True, show_default=True, help="Also write half maps + FSC.")
@click.option("--tilt", "tilt_files", multiple=True, help="Warp .xml or .tlt per tomogram (repeatable) -> wedge-weighted average.")
@click.option("--wedge-sign", type=click.Choice(["auto", "1", "-1"]), default="auto", show_default=True,
              help="Tilt-angle sign; auto = -1 for Warp XML with AreAnglesInverted, else +1. Check with 'py3sta wedge-check'.")
@click.option("--dose-weight", is_flag=True, help="Weight by per-tilt exposure (Warp XML <Dose>).")
@click.option("--backend", type=click.Choice(["numpy", "torch"]), default="numpy", show_default=True)
@click.option("-j", "--jobs", type=int, default=0, help="CPU worker processes for the numpy backend (0 = all cores).")
@click.option("--gpu", default="0", show_default=True,
              help="GPU for --backend torch: index (0, 1, ...) or 'cpu'. Indices follow CUDA_VISIBLE_DEVICES.")
def average(star, tomograms, box, tomo_dir, ts_apix, output, no_rotate, no_normalize, halves, tilt_files, wedge_sign,
            dose_weight, backend, jobs, gpu):
    """Extract particles from tomograms and write the global average.

    Tomograms are read from the STAR file's rlnMicrographName (paths as written,
    or relative to the STAR file, or inside --tomo-dir). Listing TOMOGRAMS on the
    command line overrides that.
    """
    import numpy as np

    from .average import average_particles, fsc, map_tomograms, resolution_at, tomograms_from_star
    from .io.mrc import write_mrc

    ps = ParticleSet.from_star(star, ts_apix)
    try:
        tomo_map = map_tomograms(ps, tomograms) if tomograms else tomograms_from_star(ps, star, tomo_dir)
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
    for n, p in tomo_map.items():
        click.echo(f"tomogram {n}: {p}")
    missing = set(ps.df.rlnTomoName) - set(tomo_map)
    if missing:
        click.echo(f"warning: no tomogram for {sorted(missing)}; those particles are skipped")
        ps.df = ps.df[ps.df.rlnTomoName.isin(list(tomo_map))].reset_index(drop=True)

    def prog(i, n):
        if i % 50 == 0 or i == n:
            click.echo(f"  {i}/{n}")

    tilts = _load_tilts(tilt_files, wedge_sign, list(tomo_map)) if tilt_files else None
    kw = dict(rotate=not no_rotate, normalize=not no_normalize, progress=prog, tilts=tilts, dose_weight=dose_weight)
    if backend == "torch":
        import torch

        from .backend_torch import average_particles_torch

        if gpu == "cpu":
            device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                raise click.ClickException("CUDA is not available to PyTorch; use --gpu cpu or the numpy backend")
            n = torch.cuda.device_count()
            if not gpu.isdigit() or int(gpu) >= n:
                raise click.ClickException(f"--gpu {gpu}: {n} GPU(s) visible (indices 0..{n - 1})")
            device = torch.device(f"cuda:{gpu}")
        click.echo(f"torch backend on {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
        res = average_particles_torch(ps, tomo_map, box, device=device, **kw)
    else:
        import os

        n = jobs or os.cpu_count() or 1
        click.echo(f"numpy backend, {n} worker process(es)")
        res = average_particles(ps, tomo_map, box, jobs=n, **kw)
    write_mrc(output, res.average, res.apix)
    click.echo(f"averaged {res.n_used} particles ({res.n_skipped} skipped near edges) -> {output}")
    if halves:
        stem = Path(output).with_suffix("")
        for k, h in enumerate(res.halves, 1):
            write_mrc(f"{stem}_half{k}.mrc", h, res.apix)
        r, f = fsc(*res.halves, res.apix)
        np.savetxt(f"{stem}_fsc.txt", np.column_stack([r, f]), fmt="%.4f", header="resolution_A FSC")
        click.echo(f"half-map FSC=0.143 (unmasked): first crossing {resolution_at(r, f):.1f} A, "
                   f"last crossing {resolution_at(r, f, mode='last'):.1f} A (dips at CTF zeros if not CTF-corrected)")


def _load_tilts(files, sign, tomo_names):
    from .io.particles import tomo_name_from
    from .wedge import TiltInfo

    sgn = None if sign == "auto" else float(sign)
    out = {}
    for f in files:
        name = tomo_name_from(Path(f).name.removesuffix(".xml").removesuffix(".tlt"))
        if f.endswith(".xml"):
            out[name] = TiltInfo.from_warp_xml(f, sgn)
        else:
            out[name] = TiltInfo.from_tlt(f, sgn if sgn is not None else 1.0)
    if len(files) == 1 and len(tomo_names) == 1:
        out = {tomo_names[0]: next(iter(out.values()))}
    for n, t in out.items():
        click.echo(f"wedge {n}: {len(t.angles)} tilts, coverage {t.range[0]:.1f}..{t.range[1]:.1f} deg (sign {t.sign:+.0f})")
    return out


@main.command("wedge-check")
@click.argument("tomogram")
@click.argument("tilt_file")
@click.option("--n", type=int, default=200, show_default=True, help="Random boxes to sample.")
def wedge_check(tomogram, tilt_file, n):
    """Measure which tilt-angle sign matches the tomogram's actual Fourier coverage."""
    from .io.mrc import read_mrc
    from .wedge import TiltInfo, measure_wedge_sign

    t = TiltInfo.from_warp_xml(tilt_file) if tilt_file.endswith(".xml") else TiltInfo.from_tlt(tilt_file)
    vol, _ = read_mrc(tomogram)
    r = measure_wedge_sign(vol, t, n=n)
    click.echo(f"score sign +1: {r['scores'][1.0]:.3f}   sign -1: {r['scores'][-1.0]:.3f}   -> use --wedge-sign {r['best_sign']:+.0f}")


if __name__ == "__main__":
    main()
