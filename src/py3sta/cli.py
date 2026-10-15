"""Command line: ``py3sta <command> --help``."""
from __future__ import annotations

import glob
import shutil
from pathlib import Path

import click

from .io.dipole import dipoles_to_particles
from .io.particles import DEFAULT_TOMO_REGEX, ParticleSet
from .utils.add_warp_xml import WARP_XML_COLUMN


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
@click.argument("project_dir")
@click.option("--star", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Particle STAR file (tomograms in rlnMicrographName, tilts in rlnWarpXMLfile).")
@click.option("--box", type=int, help="Box size in tomogram voxels. Default: rlnImageSize from data_optics.")
@click.option("--ctf", "ctf_mode", type=click.Choice(["none", "flip", "wiener", "amplitude"]), default="none",
              show_default=True, help="CTF correction (decide with 'py3sta ctf-check').")
@click.option("--wedge/--no-wedge", default=None, help="Wedge weighting. Default: on if rlnWarpXMLfile is present.")
@click.option("--backend", type=click.Choice(["numpy", "torch"]), default="numpy", show_default=True)
@click.option("--gpu", default="0", show_default=True, help="GPU index, or 'cpu' (torch backend).")
@click.option("-j", "--jobs", type=int, default=0, show_default=True, help="CPU worker processes (0 = all cores).")
@click.option("--particle-radius", type=float, help="Radius (Å) of the per-particle normalisation sphere. Default: half the box.")
@click.option("--margin", type=int, default=4, show_default=True, help="Stack crop margin (voxels per side; drift allowance).")
@click.option("--force", is_flag=True, help="Overwrite project.yaml, input.star and cycle-000/{cycle.yaml,00_input.star}.")
def init(project_dir, star, box, ctf_mode, wedge, backend, gpu, jobs, particle_radius, margin, force):
    """Create a project folder (project.yaml, input.star, cycle-000/) from a particle STAR file."""
    from .pipeline import ConfigError, init_project

    try:
        rep = init_project(project_dir, star, box=box, ctf_mode=ctf_mode, wedge=wedge, backend=backend,
                           gpu=int(gpu) if gpu.isdigit() else gpu, jobs=jobs, force=force,
                           particle_radius=particle_radius, margin=margin)
    except (ConfigError, ValueError, FileNotFoundError, NotImplementedError) as e:
        raise click.ClickException(str(e))
    cfg = rep.project.config
    click.echo(f"{rep.n_particles} particles in {len(rep.per_tomo)} tomograms")
    for t, n in rep.per_tomo.items():
        rng = rep.tilt_ranges.get(t)
        click.echo(f"  {t:40s} {n:6d}" + (f"   tilts {rng[0]:.1f}..{rng[1]:.1f} deg" if rng else ""))
    click.echo(f"box {cfg.box} voxels at {cfg.apix:g} Å ({cfg.box_angst:g} Å); coordinates at {cfg.ts_apix:g} Å; "
               f"normalisation radius {cfg.norm_radius_vox * cfg.apix:g} Å; wedge {'on' if cfg.wedge.use else 'off'}; "
               f"ctf {ctf_mode}; {cfg.compute.backend} backend")
    for w in rep.warnings:
        click.echo(f"warning: {w}")
    click.echo(f"project written to {rep.project.root} (edit cycle-000/cycle.yaml, then 'py3sta run {project_dir} 0')")


def _cycle_number(cycle) -> int:
    import re

    m = re.fullmatch(r"(?:cycle-)?(\d+)", str(cycle).rstrip("/").split("/")[-1])
    if not m:
        raise click.BadParameter(f"{cycle!r}: give a cycle number or folder name (0, cycle-000)")
    return int(m.group(1))


@main.command()
@click.argument("project_dir")
@click.argument("cycle")
@click.option("--from", "start", help="First step (default: the first one).")
@click.option("--to", "stop", help="Last step (default: the last implemented one).")
@click.option("--force", is_flag=True, help="Run even if an upstream step is stale; re-run over hand-edited STAR outputs (backups kept).")
def run(project_dir, cycle, start, stop, force):
    """Run the steps of a cycle (refreshes the particle stack first). CYCLE: 0 or cycle-000."""
    from .pipeline import ConfigError, load_project
    from .pipeline.manifest import StepError
    from .pipeline.runner import run_cycle

    try:
        run_cycle(load_project(project_dir), _cycle_number(cycle), start, stop, force, log=click.echo)
    except (ConfigError, StepError, KeyError, ValueError, FileNotFoundError, RuntimeError) as e:
        raise click.ClickException(str(e))


@main.command("align-maps")
@click.argument("fixed")
@click.argument("moving", nargs=-1, required=True)
@click.option("-o", "--out-dir", default="aligned_maps", show_default=True)
@click.option("--mask", "mask_file", help="Mask MRC in the fixed map's box (default: soft sphere, 40 % of the box).")
@click.option("--mask-radius", type=float, help="Sphere mask radius (Å).")
@click.option("--mask-soft", type=float, help="Soft edge width (Å).")
@click.option("--lowpass", type=float, default=30.0, show_default=True, help="Low-pass for the search (Å).")
@click.option("--highpass", type=float, help="High-pass for the search (Å).")
@click.option("--bin", "bin_factor", type=int, default=2, show_default=True, help="Binning for the global search.")
@click.option("--step", type=float, default=10.0, show_default=True, help="Global angular step (deg).")
@click.option("--max-shift", type=float, default=30.0, show_default=True, help="Shift search range (Å).")
@click.option("--polish/--no-polish", default=True, show_default=True,
              help="Refine rotation + sub-voxel shift continuously (Nelder-Mead) after the grid search.")
@click.option("--no-mirror", is_flag=True, help="Do not search the mirrored map (hand check).")
@click.option("--invert", is_flag=True, help="Flip the contrast of the moving maps.")
@click.option("--center", is_flag=True, help="Centre the moving maps on their density (centre of mass) first.")
@click.option("--backend", type=click.Choice(["numpy", "torch"]), default="torch", show_default=True)
@click.option("--gpu", default="0", show_default=True, help="GPU index, or 'cpu' (torch backend).")
def align_maps_cmd(fixed, moving, out_dir, mask_file, mask_radius, mask_soft, lowpass, highpass, bin_factor, step,
                   max_shift, polish, no_mirror, invert, center, backend, gpu):
    """Align MOVING map(s) onto FIXED: global rotation + shift search, optional mirror, polish.

    Moving maps are resampled to the fixed map's voxel size and box (EMDB maps work directly).
    Writes <name>_aligned.mrc, <name>_fsc.txt and alignment.star (RELION angles = pose of the fixed
    structure in the moving map, shift in Å)."""
    from types import SimpleNamespace

    from .align import align_map_files
    from .pipeline.steps import resolve_device

    try:
        device, label = resolve_device(SimpleNamespace(backend=backend, gpu=gpu if gpu == "cpu" else int(gpu), jobs=0))
    except RuntimeError as e:
        raise click.ClickException(str(e))
    click.echo(f"device: {label}")
    try:
        align_map_files(fixed, moving, out_dir, mask_file, mask_radius, mask_soft, invert, center, device,
                        log=click.echo, lowpass=lowpass, highpass=highpass, bin_factor=bin_factor, step=step,
                        max_shift_A=max_shift, polish=polish, try_mirror=not no_mirror)
    except (ValueError, FileNotFoundError, NotImplementedError) as e:
        raise click.ClickException(str(e))
    click.echo(f"wrote {out_dir}/alignment.star")


@main.command()
@click.argument("project_dir")
def stack(project_dir):
    """Create or update the particle stack for the latest cycle (normally done by 'run')."""
    from .pipeline import ConfigError, load_project
    from .pipeline.runner import refresh_project_stack
    from .stack import StackError

    proj = load_project(project_dir)
    if not proj.cycles:
        raise click.ClickException("no cycle folder")
    try:
        refresh_project_stack(proj, proj.cycles[-1], log=click.echo)
    except (ConfigError, StackError, FileNotFoundError) as e:
        raise click.ClickException(str(e))


@main.command()
@click.argument("project_dir", default=".")
def status(project_dir):
    """Show the state of every cycle and step (done / stale / missing / running / failed / edited)."""
    from .pipeline import ConfigError, load_project
    from .pipeline.runner import project_status

    try:
        proj = load_project(project_dir)
    except ConfigError as e:
        raise click.ClickException(str(e))
    st = project_status(proj)
    if st.stack_rows:
        line = f"stack: {st.stack_rows} particles, {st.stack_bytes / 1e9:.1f} GB"
        r = st.stack_report
        if r:
            line += (f"; last refresh {r['time']} (cycle {r['cycle']}): {r['n_new']} new, "
                     f"{r['n_tomogram']} tomogram changed, {r['n_drift']} drift")
        if st.drift_warning:
            line += f"; {st.drift_warning} particle(s) beyond half the drift allowance"
        click.echo(line)
    else:
        click.echo("stack: not built yet")
    for cs in st.cycles:
        head = f"cycle-{cs.n:03d}"
        if cs.error:
            click.echo(f"{head}  input problem: {cs.error}")
        else:
            click.echo(f"{head}  {cs.n_particles} particles ({cs.n_excluded} excluded), ctf {cs.ctf_mode}")
        for s, state in cs.steps:
            if state.state == "not implemented":
                continue
            m, extra = state.manifest, ""
            if m and m.get("status") == "done":
                i = m.get("info", {})
                extra = f"  {m['finished']}  {m['seconds']:.0f} s  {m.get('device') or ''}"
                if "fsc143_last" in i:
                    extra += f"  FSC0.143 {i['fsc143_last']} Å"
                if i.get("low_cc_tomograms"):
                    extra += f"  low CC: {', '.join(i['low_cc_tomograms'])}"
            click.echo(f"  {s.folder:14s} {state.describe()}{extra}")
        pending = [s.folder for s, state in cs.steps if state.state == "not implemented"]
        if pending:
            click.echo(f"  (not implemented yet: {', '.join(pending)})")


@main.command("next")
@click.argument("project_dir")
@click.argument("cycle")
@click.option("--force", is_flag=True, help="Copy the final STAR even if the final step is stale; replace an existing next cycle's cycle.yaml and 00_input.star.")
def next_cmd(project_dir, cycle, force):
    """Start the cycle after CYCLE: copy cycle.yaml (edit it before running) and the final STAR as 00_input.star."""
    from .pipeline import ConfigError, load_project
    from .pipeline.project import cycle_dirname, next_cycle

    try:
        rep = next_cycle(load_project(project_dir), _cycle_number(cycle), force)
    except ConfigError as e:
        raise click.ClickException(str(e))
    for w in rep.warnings:
        click.echo(f"warning: {w}")
    click.echo(f"{cycle_dirname(rep.n)}: {rep.n_particles} particles from {rep.source.name}; "
               f"edit {cycle_dirname(rep.n)}/cycle.yaml, then 'py3sta run {project_dir} {rep.n}'")


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
@click.option("--ts-apix", type=float, help="Pixel size of STAR coordinates if not in the file (rlnTomoTiltSeriesPixelSize/rlnImagePixelSize).")
@click.option("-o", "--output", default="average.mrc", show_default=True)
@click.option("--no-rotate", is_flag=True, help="Ignore angles (raw, unaligned sum).")
@click.option("--no-normalize", is_flag=True, help="Don't normalise each subvolume to mean 0 / std 1.")
@click.option("--halves/--no-halves", default=True, show_default=True, help="Also write half maps + FSC.")
@click.option("--tilt", "tilt_files", multiple=True, help="Warp .xml or .tlt per tomogram (repeatable); overrides rlnWarpXMLfile.")
@click.option("--no-wedge", is_flag=True, help="Ignore tilt information (plain real-space average).")
@click.option("--wedge-sign", type=click.Choice(["auto", "1", "-1"]), default="auto", show_default=True,
              help="Tilt-angle sign; auto = -1 for Warp XML with AreAnglesInverted, else +1. Check with 'py3sta wedge-check'.")
@click.option("--dose-weight", is_flag=True, help="Weight by per-tilt exposure (Warp XML <Dose>).")
@click.option("--ctf", "ctf_mode", type=click.Choice(["none", "flip", "wiener", "amplitude"]), default="none",
              show_default=True,
              help="CTF correction from the Warp XML. flip/wiener: tomograms WITHOUT CTF correction (IMOD/etomo WBP). "
                   "amplitude: already phase-flipped tomograms (Warp, novaCTF). Check with 'py3sta ctf-check'.")
@click.option("--ctf-height-sign", type=float, default=1.0, show_default=True,
              help="Direction of the per-particle defocus offset with height (1, -1, or 0 to ignore height).")
@click.option("--particle-radius", type=float, help="Radius (Å) of the per-particle normalisation sphere. Default: half the box.")
@click.option("--invert-contrast", is_flag=True, help="Multiply densities by -1 (protein white on black, for WBP tomograms).")
@click.option("--backend", type=click.Choice(["numpy", "torch"]), default="numpy", show_default=True)
@click.option("-j", "--jobs", type=int, default=0, help="CPU worker processes for the numpy backend (0 = all cores).")
@click.option("--gpu", default="0", show_default=True,
              help="GPU for --backend torch: index (0, 1, ...) or 'cpu'. Indices follow CUDA_VISIBLE_DEVICES.")
def average(star, tomograms, box, ts_apix, output, no_rotate, no_normalize, halves, tilt_files, wedge_sign,
            dose_weight, particle_radius, invert_contrast, backend, jobs, gpu, no_wedge, ctf_mode, ctf_height_sign):
    """Extract particles from tomograms and write the global average.

    Tomograms are read from the STAR file's rlnMicrographName (paths as written,
    or relative to the STAR file). Listing TOMOGRAMS on the command line overrides that.

    Wedge weighting uses the Warp XML in rlnWarpXMLfile (add it with 'py3sta add-xml')
    or the files given with --tilt; --no-wedge turns it off.
    """
    import numpy as np

    from .average import average_particles, fsc, map_tomograms, resolution_at, tomograms_from_star
    from .io.mrc import write_mrc

    ps = ParticleSet.from_star(star, ts_apix)
    try:
        tomo_map = map_tomograms(ps, tomograms) if tomograms else tomograms_from_star(ps, star)
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

    tilts = None
    if not no_wedge:
        if tilt_files:
            tilts = _load_tilts(tilt_files, wedge_sign, list(tomo_map))
        elif WARP_XML_COLUMN in ps.df:
            tilts = _tilts_from_star(ps, star, wedge_sign, list(tomo_map))
    ctf_mode = None if ctf_mode == "none" else ctf_mode
    if tilts is None:
        if dose_weight or ctf_mode:
            raise click.ClickException("--dose-weight / --ctf need tilt info (rlnWarpXMLfile or --tilt)")
        click.echo("no tilt info: plain average (add rlnWarpXMLfile with 'py3sta add-xml' for wedge weighting)")
    if ctf_mode:
        no_ctf = [t for t, v in tilts.items() if v.ctf is None]
        if no_ctf:
            raise click.ClickException(f"--ctf: no CTF information for {no_ctf} (need Warp XML, not .tlt)")
        click.echo(f"CTF correction: {ctf_mode} (height sign {ctf_height_sign:+g})")
    from .io.mrc import read_header

    norm_radius = None
    if particle_radius:
        norm_radius = particle_radius / read_header(next(iter(tomo_map.values())))["apix"]
    kw = dict(rotate=not no_rotate, normalize=not no_normalize, progress=prog, tilts=tilts, dose_weight=dose_weight,
              ctf_mode=ctf_mode, height_sign=ctf_height_sign, norm_radius=norm_radius, invert=invert_contrast)
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


def _report_tilts(out):
    for n, t in out.items():
        click.echo(f"wedge {n}: {len(t.angles)} tilts, coverage {t.range[0]:.1f}..{t.range[1]:.1f} deg (sign {t.sign:+.0f})")


def _tilts_from_star(ps, star, sign, tomo_names):
    from .average import per_tomo_files
    from .wedge import TiltInfo

    sgn = None if sign == "auto" else float(sign)
    try:
        files = per_tomo_files(ps, star, WARP_XML_COLUMN)
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
    out = {t: TiltInfo.from_warp_xml(files[t], sgn) for t in tomo_names if t in files}
    _report_tilts(out)
    return out


def _load_tilts(files, sign, tomo_names):
    from .io.particles import tomo_key
    from .wedge import TiltInfo

    sgn = None if sign == "auto" else float(sign)
    out = {}
    for f in files:
        name = tomo_key(Path(f).name.removesuffix(".tlt"))
        if f.endswith(".xml"):
            out[name] = TiltInfo.from_warp_xml(f, sgn)
        else:
            out[name] = TiltInfo.from_tlt(f, sgn if sgn is not None else 1.0)
    if len(files) == 1 and len(tomo_names) == 1:
        out = {tomo_names[0]: next(iter(out.values()))}
    _report_tilts(out)
    return out


@main.command("add-xml")
@click.argument("star")
@click.argument("xml", nargs=-1, required=True)
@click.option("-o", "--output", help="Output STAR (default: <input>_xml.star).")
@click.option("--relative", is_flag=True, help="Write paths relative to the output STAR file (default: absolute).")
@click.option("--allow-missing", is_flag=True, help="Leave tomograms without an XML empty instead of failing.")
def add_xml(star, xml, output, relative, allow_missing):
    """Add rlnWarpXMLfile to a particle STAR file.

    XML can be files, directories or glob patterns. Each particle gets the Warp
    tilt-series XML whose name matches its tomogram (TS_01.mrc.xml <-> TS_01).
    """
    from .utils.add_warp_xml import attach_warp_xml

    out = output or str(Path(star).with_suffix("")) + "_xml.star"
    try:
        m = attach_warp_xml(star, xml, out, relative, allow_missing)
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
    for t, p in m.items():
        click.echo(f"{t}: {p}")
    click.echo(f"{len(m)} tomogram(s) matched -> {out}")


@main.command("ctf-check")
@click.argument("star")
@click.argument("reference")
@click.option("--box", type=int, required=True)
@click.option("-j", "--jobs", type=int, default=0, help="CPU worker processes (0 = all cores).")
@click.option("--max-particles", type=int, default=300, show_default=True, help="Use at most this many particles.")
def ctf_check(star, reference, box, jobs, max_particles):
    """Is the tomogram already CTF-corrected? Which --ctf mode to use?

    Averages the particles with and without phase flipping and compares each to
    a CTF-corrected REFERENCE map (same box/pixel, aligned: e.g. a RELION/M
    refinement) between the first and second CTF zero, where an uncorrected
    tomogram has inverted phases. Needs rlnWarpXMLfile and rlnMicrographName.
    Note: half-map FSC cannot answer this; both half maps share the same CTF.
    """
    import os

    import numpy as np

    from .average import average_particles, fsc, tomograms_from_star
    from .io.mrc import read_mrc
    from .wedge import electron_wavelength

    ps = ParticleSet.from_star(star)
    if len(ps) > max_particles:
        ps.df = ps.df.sample(max_particles, random_state=0).reset_index(drop=True)
    tomo_map = tomograms_from_star(ps, star)
    tilts = _tilts_from_star(ps, star, "auto", list(tomo_map))
    ref, _ = read_mrc(reference, mmap=False)
    if ref.shape != (box,) * 3:
        raise click.ClickException(f"reference is {ref.shape}, expected box {box}")
    t0 = next(iter(tilts.values()))
    apix = None
    n = jobs or os.cpu_count() or 1
    df = float(np.median(t0.ctf.defocus_um)) * 1e4
    lam = electron_wavelength(t0.ctf.voltage_kv)
    z1, z2 = 1 / np.sqrt(1 / (lam * df)), 1 / np.sqrt(2 / (lam * df))
    r = np.sqrt(sum(g ** 2 for g in np.meshgrid(*[np.arange(box) - box // 2] * 3, indexing="ij")))
    mask = 0.5 - 0.5 * np.cos(np.pi * np.clip((0.42 * box - r) / 4, 0, 1))
    click.echo(f"median defocus {df / 1e4:.2f} um -> CTF zeros at {z1:.1f} and {z2:.1f} A; {len(ps)} particles")
    out = {}
    for mode in (None, "flip"):
        res = average_particles(ps, tomo_map, box, tilts=tilts, ctf_mode=mode, jobs=n)
        apix = res.apix
        rr, f = fsc(res.average * mask, ref * mask, apix)
        f = f * np.sign(f[rr > 1.3 * z1].mean())          # ignore global contrast inversion
        out[mode] = float(f[(rr < 0.97 * z1) & (rr > 1.03 * z2)].mean())
        click.echo(f"  {'no correction' if mode is None else 'phase flipped':14s}: correlation with reference "
                   f"between the zeros {out[mode]:+.2f}")
    if out[None] > 0 > out["flip"]:
        click.echo("-> tomogram is already phase-flipped (e.g. Warp, novaCTF): use --ctf none or --ctf amplitude")
    elif out["flip"] > 0 > out[None]:
        click.echo("-> tomogram is NOT CTF-corrected: use --ctf flip (or wiener)")
    else:
        click.echo("-> inconclusive (low resolution or reference/alignment mismatch)")


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
