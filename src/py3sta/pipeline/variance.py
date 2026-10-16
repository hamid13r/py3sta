"""``py3sta variance``: voxel-wise variance of a cycle's aligned particles (variance.py) on demand, to see
where the stack is heterogeneous before drawing MSA masks. Not a pipeline step: results go to
``cycle-NNN/variance/<name>/`` (or ``out``) and are not tracked."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ConfigError


@dataclass
class VarianceReport:
    out: Path
    n_particles: int
    n_excluded: int
    poses: Path
    mode: str
    stats: dict
    warnings: list = field(default_factory=list)


def _load_mask(project, n: int, spec: str, what: str):
    from ..io.mrc import read_mrc
    from ..reference import resolve_file
    from .steps import get_step

    p = project.config
    if spec == "reference":
        path = project.cycle_dir(n) / get_step("reference").folder / "mask.mrc"
        if not path.is_file():
            raise FileNotFoundError(f"{path}: run the reference step first (or give an MRC file for {what})")
    else:
        path = resolve_file(spec, project.root)
    vol, apix = read_mrc(path, mmap=False)
    vol = np.asarray(vol, np.float32)
    if vol.shape != (p.box,) * 3:
        raise ConfigError(f"{what} {path}: shape {vol.shape} is not the project box {p.box}^3")
    warn = None
    if apix and abs(apix - p.apix) > 0.01 * p.apix:
        warn = f"{what} {path.name}: voxel size {apix:g} Å in the header, project {p.apix:g} Å (used as is)"
    return vol, path, warn


def run_variance(project, n: int, norm_mask: str, mask: str | None = None, mode: str | None = None,
                 wedge: bool | None = None, bin: int | None = None, lowpass: float | None = None,
                 highpass: float | None = None, star=None, name: str | None = None, out=None,
                 log=print) -> VarianceReport:
    """Variance map of cycle ``n``'s active particles at the poses MSA would use (or ``star``).
    norm_mask / mask: MRC files (project box) or "reference" (02_reference/mask.mrc); mask None = everywhere.
    mode / wedge / bin / lowpass / highpass default to the cycle's msa: section."""
    from .. import plots
    from ..io.mrc import write_mrc
    from ..msa import MSASettings
    from ..stack import ParticleStack
    from ..variance import region_stats, variance_map
    from ..wedge import TiltInfo
    from .project import load_cycle_input
    from .steps import pose_star, resolve_device

    p = project.config
    c = project.cycle_config(n).msa
    s = MSASettings(mode=mode or c.mode, wedge=c.wedge if wedge is None else wedge, bin=bin or c.bin,
                    lowpass=c.lowpass if lowpass is None else (lowpass or None),
                    highpass=c.highpass if highpass is None else (highpass or None))
    if p.box % s.bin:
        raise ConfigError(f"box {p.box} is not divisible by bin {s.bin}")
    if s.lowpass and s.lowpass < 2 * p.apix * s.bin:
        raise ConfigError(f"lowpass {s.lowpass:g} Å is beyond Nyquist of bin {s.bin} ({2 * p.apix * s.bin:g} Å)")
    if s.uses_wedge and not p.wedge.use:
        raise ConfigError(f"mode {s.mode}{' + wedge' if s.mode == 'plain' else ''} needs tilt information "
                          "(wedge.use in project.yaml); use --mode plain without --wedge")
    if star is None:
        star, src = pose_star(project, n, "msa")
        if not Path(star).is_file():
            raise FileNotFoundError(f"{star}: the poses MSA would use ({src}) do not exist yet; run that step "
                                    "or give --star")
    star = Path(star)
    warnings = []
    nm, nm_path, w = _load_mask(project, n, norm_mask, "normalisation mask")
    warnings += [w] if w else []
    vm = vm_path = None
    if mask:
        vm, vm_path, w = _load_mask(project, n, mask, "variance mask")
        warnings += [w] if w else []
    cin = load_cycle_input(project, n, star)
    tilts = None
    if s.uses_wedge:
        sign = None if p.wedge.sign == "auto" else float(p.wedge.sign)
        tilts = {t: TiltInfo.from_warp_xml(f, sign) for t, f in cin.tilt_files.items()}
    device, label = resolve_device(p.compute)
    if name is None:
        name = nm_path.stem.replace(".", "_") + (f"_in_{vm_path.stem.replace('.', '_')}" if vm_path else "") + f"_{s.mode}"
    out = Path(out) if out else project.cycle_dir(n) / "variance" / name
    out.mkdir(parents=True, exist_ok=True)
    what = {"wmd": "wedge-masked differences", "plain": "plain" + (" (own wedge)" if s.wedge else "")}[s.mode]
    log(f"variance of {len(cin.particles)} particles ({cin.n_excluded} excluded) from {star.name}: {what}, "
        f"bin {s.bin}, lowpass {s.lowpass} Å, normalised inside {nm_path.name}; {label}")
    step = max(1, len(cin.particles) // 5)
    last = [0]

    def progress(done, total):
        if done - last[0] >= step or done == total:
            last[0] = done
            log(f"  {done}/{total}")

    res = variance_map(cin.particles, ParticleStack.open(project.stack_dir), nm, p.apix, s, tilts,
                       invert=p.invert_contrast, device=device, progress=progress)
    write_mrc(out / "variance.mrc", res.variance, res.apix)
    write_mrc(out / "std.mrc", np.sqrt(res.variance), res.apix)
    write_mrc(out / "average.mrc", res.average, res.apix)
    if s.mode == "wmd":
        write_mrc(out / "mean_difference.mrc", res.mean, res.apix)
    vols, labels = [res.average, res.variance], ["average", "variance"]
    if vm is not None:
        from ..msa import binned_mask

        masked = res.variance * (binned_mask(vm, p.apix, s.bin) > 0.5)
        write_mrc(out / "variance_masked.mrc", masked.astype(np.float32), res.apix)
        vols.append(masked)
        labels.append("in variance mask")
    stats = region_stats(res, nm, p.apix, s.bin, vm)
    info = {"n_particles": res.n, "n_excluded": cin.n_excluded, "poses": str(star), "mode": s.mode,
            "wedge": s.wedge, "bin": s.bin, "lowpass": s.lowpass, "highpass": s.highpass, "box": res.box,
            "apix": res.apix, "norm_mask": str(nm_path), "variance_mask": str(vm_path) if vm_path else None,
            "stats": stats, "warnings": warnings}
    (out / "stats.json").write_text(json.dumps(info, indent=1))
    if plots.available():
        title = f"{project.root.name} cycle {n:03d}: {what}, normalised in {nm_path.name}"
        plots.volume_views(out / "variance.png", vols, labels, title=title + " (projections)")
        plots.volume_views(out / "variance_slices.png", vols, labels, title=title + " (central slices)", project=False)
    else:
        warnings.append("matplotlib not installed: no PNGs")
    return VarianceReport(out, res.n, cin.n_excluded, star, s.mode, stats, warnings)
