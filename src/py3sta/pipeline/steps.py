"""Cycle steps in pipeline order (PLAN.md §2). Each step has a folder, upstream steps, the
parameters that change its result (compute settings excluded: numpy and torch agree), and a run
function ``run(ctx, project, n, cycle_cfg)``. Steps without ``run`` are not implemented yet."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from .config import CycleConfig, ProjectConfig


@dataclass(frozen=True)
class Step:
    name: str
    folder: str
    upstream: tuple = ()
    params: object = None             # (ProjectConfig, CycleConfig) -> dict
    run: object = None                # (StepContext, Project, n, CycleConfig) -> None

    @property
    def implemented(self) -> bool:
        return self.run is not None


def resolve_device(compute) -> tuple[object, str]:
    """(torch.device or None, label) for the project's compute settings."""
    if compute.backend != "torch":
        import os

        return None, f"numpy, {compute.jobs or os.cpu_count()} process(es)"
    import torch

    if compute.gpu == "cpu":
        return torch.device("cpu"), "torch cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("compute.backend is torch but CUDA is not available (set compute.gpu: cpu or backend: numpy)")
    n = torch.cuda.device_count()
    if int(compute.gpu) >= n:
        raise RuntimeError(f"compute.gpu {compute.gpu}: {n} GPU(s) visible")
    dev = torch.device(f"cuda:{compute.gpu}")
    return dev, f"torch {dev} ({torch.cuda.get_device_name(dev)})"


# ----------------------------------------------------------------- average
def average_params(p: ProjectConfig, c: CycleConfig) -> dict:
    return {"average": dataclasses.asdict(c.average), "ctf": dataclasses.asdict(c.ctf),
            "wedge": dataclasses.asdict(p.wedge), "box": p.box, "apix": p.apix, "ts_apix": p.ts_apix,
            "particle_radius": p.norm_radius_vox * p.apix}


def run_average(ctx, project, n: int, c: CycleConfig) -> None:
    from ..average import fsc, per_tomo_cc, resolution_at
    from ..io.mrc import write_mrc
    from ..stack import ParticleStack
    from ..wedge import TiltInfo
    from .project import load_cycle_input

    p = project.config
    cin = load_cycle_input(project, n)
    ctx.read_star(project.cycle_input(n))
    ctx.use_files(cin.tomograms.values(), "tomogram")
    ctx.use_files(cin.tilt_files.values(), "tilt file")
    tilts = None
    if p.wedge.use:
        sign = None if p.wedge.sign == "auto" else float(p.wedge.sign)
        tilts = {t: TiltInfo.from_warp_xml(f, sign) for t, f in cin.tilt_files.items()}
        if c.ctf.mode != "none":
            no_ctf = [t for t, v in tilts.items() if v.ctf is None]
            if no_ctf:
                raise ValueError(f"ctf.mode {c.ctf.mode}: no CTF information for {', '.join(no_ctf[:5])}")
    stack = ParticleStack.open(project.stack_dir)
    kw = dict(normalize=c.average.normalize, tilts=tilts, dose_weight=p.wedge.dose_weight,
              ctf_mode=None if c.ctf.mode == "none" else c.ctf.mode, height_sign=c.ctf.height_sign,
              norm_radius=p.norm_radius_vox, stack=stack, per_tomo=True)
    device, _ = resolve_device(p.compute)
    ctx.log(f"averaging {len(cin.particles)} particles from the stack ({cin.n_excluded} excluded)")
    if device is None:
        import os

        from ..average import average_particles

        res = average_particles(cin.particles, None, p.box, jobs=p.compute.jobs or os.cpu_count(), **kw)
    else:
        from ..backend_torch import average_particles_torch

        res = average_particles_torch(cin.particles, None, p.box, device=device, **kw)
    write_mrc(ctx.output("average.mrc"), res.average, res.apix)
    info = ctx.info
    info.update(n_particles=cin.n_total, n_excluded=cin.n_excluded, n_used=res.n_used, n_skipped=res.n_skipped)
    if c.average.halves:
        for k, h in enumerate(res.halves, 1):
            write_mrc(ctx.output(f"half{k}.mrc"), h, res.apix)
        r, f = fsc(*res.halves, res.apix)
        np.savetxt(ctx.output("fsc.txt"), np.column_stack([r, f]), fmt="%.4f", header="resolution_A FSC")
        info.update(fsc143_first=round(resolution_at(r, f), 1), fsc143_last=round(resolution_at(r, f, mode="last"), 1))
    cc = per_tomo_cc(res, p.norm_radius_vox, c.average.cc_lowpass)
    cc.to_csv(ctx.output("per_tomo.csv"), index=False, float_format="%.4f")
    flagged = cc.loc[cc["flagged"], "tomo"].tolist()
    info.update(cc_median=round(float(cc["cc"].median()), 3), low_cc_tomograms=flagged)
    if flagged:
        ctx.warn(f"low per-tomogram CC: {', '.join(flagged)} (see per_tomo.csv)")


# ------------------------------------------------------------------- order
STEPS: tuple[Step, ...] = (
    Step("average", "01_average", (), average_params, run_average),
    Step("reference", "02_reference", ("average",)),
    Step("align", "03_align", ("reference",)),
    Step("msa", "04_msa", ("align",)),
    Step("classify", "05_classify", ("msa",)),
    Step("embed", "06_embed", ("classify",)),
    Step("select", "07_select", ("classify",)),
)
BY_NAME = {s.name: s for s in STEPS}


def get_step(name: str) -> Step:
    if name not in BY_NAME:
        raise KeyError(f"unknown step {name!r} (steps: {', '.join(BY_NAME)})")
    return BY_NAME[name]
