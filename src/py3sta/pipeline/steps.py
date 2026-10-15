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
    upstream: object = ()             # tuple of step names, or (CycleConfig) -> tuple (mode-dependent)
    params: object = None             # (ProjectConfig, CycleConfig) -> dict
    run: object = None                # (StepContext, Project, n, CycleConfig) -> None
    poses: str | None = None          # STAR output with updated particle poses (read by 'final')

    @property
    def implemented(self) -> bool:
        return self.run is not None

    def upstream_for(self, cycle_cfg) -> tuple:
        return tuple(self.upstream(cycle_cfg) if callable(self.upstream) else self.upstream)


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
            "particle_radius": p.norm_radius_vox * p.apix, "invert_contrast": p.invert_contrast}


def run_average(ctx, project, n: int, c: CycleConfig) -> None:
    from ..average import fsc, per_tomo_cc, protein_contrast, resolution_at
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
              norm_radius=p.norm_radius_vox, stack=stack, per_tomo=True, invert=p.invert_contrast)
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
    label, d = protein_contrast(res.average, p.norm_radius_vox)
    info.update(protein_contrast=label)
    if label == "dark":
        ctx.warn("protein appears dark (black on white)"
                 + ("; set invert_contrast: true in project.yaml for white on black" if not p.invert_contrast
                    else " although invert_contrast is true: check the tomograms"))
    cc = per_tomo_cc(res, p.norm_radius_vox, c.average.cc_lowpass)
    cc.to_csv(ctx.output("per_tomo.csv"), index=False, float_format="%.4f")
    flagged = cc.loc[cc["flagged"], "tomo"].tolist()
    info.update(cc_median=round(float(cc["cc"].median()), 3), low_cc_tomograms=flagged)
    if flagged:
        ctx.warn(f"low per-tomogram CC: {', '.join(flagged)} (see per_tomo.csv)")


# --------------------------------------------------------------- reference
def reference_upstream(c: CycleConfig) -> tuple:
    return ("average",) if c.reference.mode == "global" else ()


def reference_params(p: ProjectConfig, c: CycleConfig) -> dict:
    return {"reference": dataclasses.asdict(c.reference), "box": p.box, "apix": p.apix}


def run_reference(ctx, project, n: int, c: CycleConfig) -> None:
    import pandas as pd

    from ..io.mrc import open_volume_stack, read_mrc, write_mrc
    from ..io.star import write_star
    from ..reference import load_volumes, make_mask, masked_cc, prepare_reference, resolve_file

    p, r = project.config, c.reference
    m = r.mask
    mask_file = None
    if m.shape == "file":
        mask_file = resolve_file(m.file, project.root)
        ctx.use_files([mask_file], "mask")
    mask = make_mask(p.box, p.apix, m.shape, m.radius, m.soft, m.center, mask_file)
    avg_path = project.cycle_dir(n) / "01_average" / "average.mrc"
    average = read_mrc(avg_path, mmap=False)[0] if avg_path.is_file() else None

    sources = []                                     # (volume, source, file, invert)
    if r.mode == "global":
        if average is None:
            raise FileNotFoundError(f"{avg_path}: run the average step first")
        sources.append((average, "global", str(avg_path), False))
    else:
        for name in r.files:
            f = resolve_file(name, project.root)
            ctx.use_files([f], "reference file")
            vols = load_volumes(f, p.box, p.apix)
            sources += [(v, "file", f"{f}" + (f"#{k + 1}" if len(vols) > 1 else ""), r.invert) for k, v in enumerate(vols)]

    refs = np.stack([prepare_reference(v, p.apix, mask, r.lowpass, r.highpass, inv) for v, _, _, inv in sources])
    data = open_volume_stack(ctx.output("references.mrcs"), len(refs), p.box, np.float32, p.apix, mode="w+")
    data[:] = refs
    data.flush()
    write_mrc(ctx.output("mask.mrc"), mask, p.apix)
    rows = []
    avg_f = prepare_reference(average, p.apix, mask, r.lowpass, r.highpass) if average is not None else None
    for k, (ref, (_, src, f, inv)) in enumerate(zip(refs, sources), 1):
        cc = masked_cc(ref, avg_f, mask) if avg_f is not None else np.nan
        rows.append({"py3staReferenceIndex": k, "py3staReferenceSource": src, "py3staReferenceFile": f,
                     "py3staReferenceInvert": int(inv), "py3staReferenceCC": round(float(cc), 4)})
        if src == "file" and cc < 0:
            ctx.warn(f"reference {k} anti-correlates with the current average (CC {cc:.2f}): "
                     "contrast inverted? set reference.invert")
    write_star({"references": pd.DataFrame(rows)}, ctx.output_star("references.star"))
    ctx.info.update(n_references=len(refs), mask_fraction=round(float(mask.mean()), 4),
                    cc_with_average=[row["py3staReferenceCC"] for row in rows])
    ctx.log(f"{len(refs)} reference(s); CC with the current average: "
            + ", ".join(f"{row['py3staReferenceCC']:.3f}" for row in rows))


# ------------------------------------------------------------------- final
def final_name(n: int) -> str:
    return f"cycle-{n:03d}_final.star"


def pose_source(c: CycleConfig | None = None) -> Step | None:
    """The last implemented step that writes particle poses (select, else align); None = 00_input.star."""
    return next((s for s in reversed(STEPS) if s.poses and s.implemented), None)


def final_upstream(c: CycleConfig) -> tuple:
    src = pose_source(c)
    return (src.name,) if src else ()


def final_params(p: ProjectConfig, c: CycleConfig) -> dict:
    return {"cycle": c.cycle}


def _particle_block(blocks: dict):
    return blocks["particles"] if "particles" in blocks else next(iter(blocks.values()))


def run_final(ctx, project, n: int, c: CycleConfig) -> None:
    """cycle-NNN_final.star: every particle of the pose source (excluded rows included, unchanged)
    plus py3staCycle; the next cycle's 00_input.star is a copy of it."""
    from ..io.star import read_star, write_star
    from .project import PARTICLE_NAME, load_particles

    src = pose_source(c)
    path = project.cycle_input(n) if src is None else project.cycle_dir(n) / src.folder / src.poses
    ctx.read_star(path)
    load_particles(project, path)                             # validates (also after hand edits)
    blocks = read_star(path)
    df = _particle_block(blocks)
    df["py3staCycle"] = n
    if src is not None:                                       # a step must not drop particles
        names_in = _particle_block(read_star(project.cycle_input(n)))[PARTICLE_NAME].astype(str)
        lost = set(names_in) - set(df[PARTICLE_NAME].astype(str))
    else:
        lost = ()
    if lost:
        raise ValueError(f"{path.name}: {len(lost)} particle(s) of 00_input.star missing: {', '.join(sorted(lost)[:5])}")
    write_star(blocks, ctx.output_star(final_name(n)))
    ctx.info.update(n_particles=len(df), pose_source=src.name if src else "input")
    ctx.log(f"{final_name(n)}: {len(df)} particles, poses from {src.name if src else '00_input.star'}")


# ------------------------------------------------------------------- order
STEPS: tuple[Step, ...] = (
    Step("average", "01_average", (), average_params, run_average),
    Step("reference", "02_reference", reference_upstream, reference_params, run_reference),
    Step("align", "03_align", ("reference",), poses="aligned.star"),
    Step("msa", "04_msa", ("align",)),
    Step("classify", "05_classify", ("msa",)),
    Step("embed", "06_embed", ("classify",)),
    Step("select", "07_select", ("classify",), poses="selected.star"),
    Step("final", "08_final", final_upstream, final_params, run_final),
)
BY_NAME = {s.name: s for s in STEPS}


def get_step(name: str) -> Step:
    if name not in BY_NAME:
        raise KeyError(f"unknown step {name!r} (steps: {', '.join(BY_NAME)})")
    return BY_NAME[name]
