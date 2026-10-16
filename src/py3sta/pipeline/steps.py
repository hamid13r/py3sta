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


def _averager(ctx, project, c: CycleConfig, cin):
    """``average(particles, **extra) -> AverageResult`` from the stack with the cycle's wedge / CTF /
    normalisation settings on the project's device; records tomograms and tilt files as inputs."""
    from ..stack import ParticleStack
    from ..wedge import TiltInfo

    p = project.config
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
    kw = dict(normalize=c.average.normalize, tilts=tilts, dose_weight=p.wedge.dose_weight,
              ctf_mode=None if c.ctf.mode == "none" else c.ctf.mode, height_sign=c.ctf.height_sign,
              norm_radius=p.norm_radius_vox, stack=ParticleStack.open(project.stack_dir), invert=p.invert_contrast)
    device, _ = resolve_device(p.compute)

    def average(particles, **extra):
        if device is None:
            import os

            from ..average import average_particles

            return average_particles(particles, None, p.box, jobs=p.compute.jobs or os.cpu_count(), **kw, **extra)
        from ..backend_torch import average_particles_torch

        return average_particles_torch(particles, None, p.box, device=device, **kw, **extra)

    return average


def run_average(ctx, project, n: int, c: CycleConfig) -> None:
    from ..average import fsc, per_tomo_cc, protein_contrast, resolution_at
    from ..io.mrc import write_mrc
    from .project import load_cycle_input

    p = project.config
    cin = load_cycle_input(project, n)
    ctx.read_star(project.cycle_input(n))
    average = _averager(ctx, project, c, cin)
    ctx.log(f"averaging {len(cin.particles)} particles from the stack ({cin.n_excluded} excluded)")
    res = average(cin.particles, per_tomo=True)
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


# ------------------------------------------------------------- pose source
def pose_source(before: str) -> Step | None:
    """The last implemented step before ``before`` that writes particle poses; None = 00_input.star."""
    names = [s.name for s in STEPS]
    return next((s for s in reversed(STEPS[:names.index(before)]) if s.poses and s.implemented), None)


def pose_star(project, n: int, before: str):
    """(path, source step name) of the particle poses that step ``before`` starts from."""
    src = pose_source(before)
    if src is None:
        return project.cycle_input(n), "input"
    return project.cycle_dir(n) / src.folder / src.poses, src.name


def _with_pose_source(name: str, upstream=()):
    """Upstream function: fixed upstream steps + the step providing the poses."""
    def fn(c: CycleConfig) -> tuple:
        src = pose_source(name)
        ups = tuple(upstream(c) if callable(upstream) else upstream)
        return ups + ((src.name,) if src and src.name not in ups else ())
    return fn


def _source_name(before: str) -> str:
    src = pose_source(before)
    return src.name if src else "input"


def _particle_block(blocks: dict):
    return blocks["particles"] if "particles" in blocks else next(iter(blocks.values()))


# ------------------------------------------------------------------- align
ALIGN_COLUMNS = ("rlnBestReference", "py3staRefCC", "py3staAlignAngle", "py3staAlignShift", "py3staAlignEdge")
POSE_COLUMNS = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]


def align_params(p: ProjectConfig, c: CycleConfig) -> dict:
    return {"align": dataclasses.asdict(c.align), "box": p.box, "apix": p.apix, "ts_apix": p.ts_apix,
            "invert_contrast": p.invert_contrast, "wedge": dataclasses.asdict(p.wedge),
            "shift_vox": p.align_shift_vox(c.align.max_shift), "pose_source": _source_name("align")}


def update_particles(out, names, values: dict) -> None:
    """Write per-particle ``values`` {column: array aligned with ``names``} into the rows of table ``out``
    whose rlnTomoParticleName is in ``names`` (other rows untouched; new columns get 0 there)."""
    import pandas as pd

    from .project import PARTICLE_NAME

    pos = pd.Series(np.arange(len(out)), index=out[PARTICLE_NAME].astype(str)).loc[list(names)].to_numpy()
    for col, v in values.items():
        if col not in out:
            out[col] = 0
        out.loc[out.index[pos], col] = np.asarray(v)


def run_align(ctx, project, n: int, c: CycleConfig) -> None:
    """Align every active particle to the reference(s) of 02_reference (mra.py) and compose the result
    onto its pose -> aligned.star (all rows of the pose source), align.csv. ``enable: false`` passes the
    poses through unchanged."""
    import pandas as pd

    from ..io.mrc import read_mrc
    from ..io.star import read_star, write_star
    from ..mra import MRASettings, align_particles, cone_grid
    from ..poses import apply_delta
    from ..reference import make_mask, resolve_file
    from ..stack import ParticleStack
    from ..wedge import TiltInfo
    from .project import PARTICLE_NAME, load_cycle_input

    p, a = project.config, c.align
    path, src = pose_star(project, n, "align")
    ctx.read_star(path)
    blocks = read_star(path)
    out = _particle_block(blocks)
    if not a.enable:
        write_star(blocks, ctx.output_star("aligned.star"))
        ctx.info.update(enabled=False, pose_source=src)
        ctx.log("align disabled: poses passed through")
        return
    cin = load_cycle_input(project, n, path)
    ctx.use_files(cin.tomograms.values(), "tomogram")
    ref_dir = project.cycle_dir(n) / get_step("reference").folder
    refs = np.array(read_mrc(ref_dir / "references.mrcs", mmap=False)[0]).reshape(-1, p.box, p.box, p.box)
    if a.mask is None:
        mask = read_mrc(ref_dir / "mask.mrc", mmap=False)[0]
    else:
        m = a.mask
        mf = None
        if m.shape == "file":
            mf = resolve_file(m.file, project.root)
            ctx.use_files([mf], "mask")
        mask = make_mask(p.box, p.apix, m.shape, m.radius, m.soft, m.center, mf)
    tilts = None
    if a.wedge:
        ctx.use_files(cin.tilt_files.values(), "tilt file")
        sign = None if p.wedge.sign == "auto" else float(p.wedge.sign)
        tilts = {t: TiltInfo.from_warp_xml(f, sign) for t, f in cin.tilt_files.items()}
    settings = MRASettings(levels=[list(map(float, lv)) for lv in a.levels], max_shift=p.align_shift_vox(a.max_shift),
                           lowpass=a.lowpass, highpass=a.highpass, bin=a.bin, wedge=a.wedge, min_gain=a.min_gain)
    device, label = resolve_device(p.compute)
    if device is None:
        ctx.warn("align runs on the numpy reference implementation (slow); set compute.backend: torch")
    nrot = sum(len(cone_grid(*lv).mats) for lv in settings.levels)
    ps = cin.particles
    ctx.log(f"aligning {len(ps)} particles to {len(refs)} reference(s): {nrot} rotations per reference, "
            f"shift <= {settings.max_shift} voxel(s) (x, y, z), bin {a.bin}, {'constrained' if tilts else 'plain'} CC"
            + (f", min_gain {a.min_gain}" if a.min_gain else "") + f"; {label}")
    step = max(1, len(ps) // 10)
    last = [0]

    def progress(done, total):
        if done - last[0] >= step or done == total:
            last[0] = done
            ctx.log(f"  {done}/{total}")

    res = align_particles(ps, ParticleStack.open(project.stack_dir), refs, mask, p.apix, settings, tilts,
                          invert=p.invert_contrast, device=device, progress=progress)
    new = apply_delta(ps, res.R, res.t, p.apix)
    names = ps.df[PARTICLE_NAME].astype(str).to_numpy()
    shift_A = np.linalg.norm(res.t, axis=1) * p.apix
    update_particles(out, names, {**{col: new.df[col].to_numpy() for col in POSE_COLUMNS},
                                  "rlnBestReference": res.best_ref, "py3staRefCC": np.round(res.cc, 5),
                                  "py3staAlignAngle": np.round(res.angle, 3), "py3staAlignShift": np.round(shift_A, 3),
                                  "py3staAlignEdge": res.at_edge.astype(int)})
    write_star(blocks, ctx.output_star("aligned.star"))
    table = pd.DataFrame({PARTICLE_NAME: names, "rlnTomoName": ps.df["rlnTomoName"].to_numpy(),
                          "best_reference": res.best_ref, "cc": res.cc, "angle_deg": res.angle,
                          "shift_x_A": res.t[:, 0] * p.apix, "shift_y_A": res.t[:, 1] * p.apix,
                          "shift_z_A": res.t[:, 2] * p.apix, "at_edge": res.at_edge.astype(int)})
    for k in range(res.cc_ref.shape[1]):
        table[f"cc_ref{k + 1}"] = res.cc_ref[:, k]
    table.to_csv(ctx.output("align.csv"), index=False, float_format="%.4f")
    n_edge = int(res.at_edge.sum())
    counts = {str(k + 1): int((res.best_ref == k + 1).sum()) for k in range(len(refs))}
    ctx.info.update(enabled=True, n_aligned=len(ps), pose_source=src, n_rotations=nrot,
                    n_unchanged=int(((res.angle < 1e-6) & (np.abs(res.t).max(1) < 1e-6)).sum()),
                    angle_median=round(float(np.median(res.angle)), 2), shift_median_A=round(float(np.median(shift_A)), 2),
                    cc_median=round(float(np.median(res.cc)), 4), n_at_edge=n_edge, best_reference_counts=counts)
    ctx.log(f"median change {np.median(res.angle):.2f}°, {np.median(shift_A):.2f} Å; median CC {np.median(res.cc):.3f}; "
            f"at the search edge: {n_edge}" + (f"; best reference counts {counts}" if len(refs) > 1 else ""))
    if n_edge > 0.1 * len(ps):
        ctx.warn(f"{n_edge} of {len(ps)} particles at the edge of the search (first level range or shift limit): "
                 "widen align.levels[0] / align.max_shift")


# ---------------------------------------------------------------- classify
CLASS_COLUMN = "py3staClass"


def classify_upstream(c: CycleConfig) -> tuple:
    return () if c.classify.method == "column" else ("msa",)


def classify_params(p: ProjectConfig, c: CycleConfig) -> dict:
    return {"classify": dataclasses.asdict(c.classify), **average_params(p, c), "pose_source": _source_name("classify")}


def run_classify(ctx, project, n: int, c: CycleConfig) -> None:
    """method column: classes from an integer STAR column (placeholder until MSA + k-means exist);
    class averages from the stack with the cycle's averaging settings."""
    import pandas as pd

    from ..io.mrc import open_volume_stack
    from ..io.star import write_star
    from .project import PARTICLE_NAME, load_cycle_input

    p = project.config
    path, src = pose_star(project, n, "classify")
    ctx.read_star(path)
    cin = load_cycle_input(project, n, path)
    df = cin.particles.df
    col = c.classify.column
    if col not in df:
        ctx.warn(f"no column {col}: all particles in class 1")
        cls = np.ones(len(df), int)
    else:
        v = pd.to_numeric(df[col], errors="coerce").to_numpy()
        bad = ~np.isfinite(v) | (v != np.round(v))
        if bad.any():
            raise ValueError(f"{path.name}: column {col} is not an integer for {int(bad.sum())} particle(s)")
        cls = v.astype(int)
    ids = np.unique(cls)
    write_star({"classes": pd.DataFrame({PARTICLE_NAME: df[PARTICLE_NAME].astype(str), CLASS_COLUMN: cls})},
               ctx.output_star(f"class_{c.classify.method}.star"))
    average = _averager(ctx, project, c, cin)
    out = open_volume_stack(ctx.output("classes.mrcs"), len(ids), p.box, np.float32, p.apix, mode="w+")
    rows = []
    for k, cid in enumerate(ids):
        sel = cls == cid
        members = dataclasses.replace(cin.particles, df=df[sel].reset_index(drop=True))
        res = average(members)
        out[k] = res.average
        rows.append({CLASS_COLUMN: int(cid), "py3staClassIndex": k + 1, "py3staClassCount": int(sel.sum()),
                     "py3staClassFraction": round(float(sel.mean()), 4)})
        ctx.log(f"class {cid}: {int(sel.sum())} particles")
    out.flush()
    write_star({"classes": pd.DataFrame(rows)}, ctx.output_star("classes.star"))
    ctx.info.update(n_classes=len(ids), counts={int(r[CLASS_COLUMN]): r["py3staClassCount"] for r in rows},
                    pose_source=src)


# ------------------------------------------------------------------ select
def select_params(p: ProjectConfig, c: CycleConfig) -> dict:
    return {"select": dataclasses.asdict(c.select), "box": p.box, "apix": p.apix, "pose_source": _source_name("select")}


def class_actions(ids, s) -> dict:
    """{class: 'align' | 'leave' | 'exclude'}; classes not listed: align if align == 'all', else leave."""
    out = {}
    for cid in ids:
        cid = int(cid)
        if cid in s.exclude:
            out[cid] = "exclude"
        elif cid in s.leave:
            out[cid] = "leave"
        elif s.align == "all" or cid in s.align:
            out[cid] = "align"
        else:
            out[cid] = "leave"
    return out


def run_select(ctx, project, n: int, c: CycleConfig) -> None:
    """Align class averages to the target and compose each class transform onto its members (poses.py);
    leave keeps poses; exclude sets py3staExcluded. Writes selected.star (every particle of the pose source)."""
    import pandas as pd

    from ..align import align_maps, apply_alignment, load_map
    from ..io.mrc import open_volume_stack, read_mrc
    from ..io.star import read_star, write_star
    from ..poses import apply_delta
    from ..reference import resolve_file
    from .project import EXCLUDED, PARTICLE_NAME, load_cycle_input

    p, s = project.config, c.select
    cdir = project.cycle_dir(n)
    path, src = pose_star(project, n, "select")
    ctx.read_star(path)
    cin = load_cycle_input(project, n, path)
    cl_dir = cdir / get_step("classify").folder
    cls_star = cl_dir / f"class_{c.classify.method}.star"
    ctx.read_star(cls_star)
    assign = _particle_block(read_star(cls_star)).astype({PARTICLE_NAME: str}).set_index(PARTICLE_NAME)[CLASS_COLUMN]
    classes = _particle_block(read_star(cl_dir / "classes.star"))
    averages = np.array(read_mrc(cl_dir / "classes.mrcs", mmap=False)[0]).reshape(-1, p.box, p.box, p.box)

    names = cin.particles.df[PARTICLE_NAME].astype(str)
    missing = ~names.isin(assign.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} particle(s) have no class in {cls_star.name} (re-run classify)")
    cls = assign.loc[names].to_numpy(int)
    ids = classes[CLASS_COLUMN].to_numpy(int)
    actions = class_actions(ids, s)

    ref_dir = cdir / get_step("reference").folder
    mask = read_mrc(ref_dir / "mask.mrc", mmap=False)[0]
    if s.target == "reference":
        refs = read_mrc(ref_dir / "references.mrcs", mmap=False)[0].reshape(-1, p.box, p.box, p.box)
        target = np.array(refs[0])
        if len(refs) > 1:
            ctx.warn(f"{len(refs)} references: class averages are aligned to reference 1")
    else:
        f = resolve_file(s.target, project.root)
        ctx.use_files([f], "target")
        target = load_map(f, p.apix, p.box)
    device, _ = resolve_device(p.compute)

    ps = cin.particles
    rows = []
    cc_of = {}
    aligned = open_volume_stack(ctx.output("classes_aligned.mrcs"), len(ids), p.box, np.float32, p.apix, mode="w+")
    for k, cid in enumerate(ids):
        act = actions[int(cid)]
        members = cls == cid
        row = {CLASS_COLUMN: int(cid), "py3staClassAction": act, "py3staClassCount": int(members.sum()),
               "rlnAngleRot": 0.0, "rlnAngleTilt": 0.0, "rlnAnglePsi": 0.0,
               "rlnOriginXAngst": 0.0, "rlnOriginYAngst": 0.0, "rlnOriginZAngst": 0.0, "py3staClassCC": np.nan}
        aligned[k] = averages[k]
        if act == "align" and members.any():
            res = align_maps(target, averages[k], p.apix, mask, lowpass=s.lowpass, highpass=s.highpass,
                             bin_factor=s.bin, step=s.step, max_shift_A=s.max_shift, try_mirror=False, device=device)
            if res.inverted_contrast_suspected:
                ctx.warn(f"class {cid}: anti-correlates with the target (min CC {res.min_cc:.2f}): contrast inverted?")
            ps = apply_delta(ps, res.matrix, res.shift, p.apix, rows=members)
            aligned[k] = apply_alignment(averages[k], res)
            row.update(zip(["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"], np.round(res.euler, 3)))
            row.update(zip(["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"], np.round(res.shift * p.apix, 3)))
            row["py3staClassCC"] = round(res.cc, 4)
            ctx.log(f"class {cid} ({int(members.sum())}): rot/tilt/psi {np.round(res.euler, 1)}, "
                    f"shift {np.round(res.shift * p.apix, 1)} Å, CC {res.cc:.3f}")
        else:
            ctx.log(f"class {cid} ({int(members.sum())}): {act}")
        cc_of[int(cid)] = row["py3staClassCC"]
        rows.append(row)
    aligned.flush()
    write_star({"transforms": pd.DataFrame(rows)}, ctx.output_star("transforms.star"))

    # selected.star: all rows of the pose source; active rows get poses, class, score, exclusion
    blocks = read_star(path)
    out = _particle_block(blocks)
    pos = pd.Series(np.arange(len(out)), index=out[PARTICLE_NAME].astype(str)).loc[names].to_numpy()
    cols = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
    for col in cols:
        if col not in out:
            out[col] = 0.0
    out.loc[out.index[pos], cols] = ps.df[cols].to_numpy()
    for col, default in (("rlnClassNumber", 0), ("py3staClassCC", np.nan), (EXCLUDED, 0)):
        if col not in out:
            out[col] = default
    out.loc[out.index[pos], "rlnClassNumber"] = cls
    out.loc[out.index[pos], "py3staClassCC"] = [cc_of[int(k)] for k in cls]
    excl = np.array([actions[int(k)] == "exclude" for k in cls])
    if excl.any():
        if "py3staExcludedCycle" not in out:
            out["py3staExcludedCycle"] = -1
        out.loc[out.index[pos[excl]], EXCLUDED] = 1
        out.loc[out.index[pos[excl]], "py3staExcludedCycle"] = n
    write_star(blocks, ctx.output_star("selected.star"))
    ctx.info.update(actions={str(k): v for k, v in actions.items()}, n_excluded_now=int(excl.sum()),
                    class_cc={str(k): v for k, v in cc_of.items()}, pose_source=src)


# ------------------------------------------------------------------- final
def final_name(n: int) -> str:
    return f"cycle-{n:03d}_final.star"


def final_params(p: ProjectConfig, c: CycleConfig) -> dict:
    return {"cycle": c.cycle, "pose_source": _source_name("final")}


def run_final(ctx, project, n: int, c: CycleConfig) -> None:
    """cycle-NNN_final.star: every particle of the pose source (excluded rows included, unchanged)
    plus py3staCycle; the next cycle's 00_input.star is a copy of it."""
    from ..io.star import read_star, write_star
    from .project import PARTICLE_NAME, load_particles

    path, src = pose_star(project, n, "final")
    ctx.read_star(path)
    load_particles(project, path)                             # validates (also after hand edits)
    blocks = read_star(path)
    df = _particle_block(blocks)
    df["py3staCycle"] = n
    lost = ()
    if src != "input":                                        # a step must not drop particles
        names_in = _particle_block(read_star(project.cycle_input(n)))[PARTICLE_NAME].astype(str)
        lost = set(names_in) - set(df[PARTICLE_NAME].astype(str))
    if lost:
        raise ValueError(f"{path.name}: {len(lost)} particle(s) of 00_input.star missing: {', '.join(sorted(lost)[:5])}")
    write_star(blocks, ctx.output_star(final_name(n)))
    ctx.info.update(n_particles=len(df), pose_source=src)
    ctx.log(f"{final_name(n)}: {len(df)} particles, poses from {'00_input.star' if src == 'input' else src}")


# ------------------------------------------------------------------- order
STEPS: tuple[Step, ...] = (
    Step("average", "01_average", (), average_params, run_average),
    Step("reference", "02_reference", reference_upstream, reference_params, run_reference),
    Step("align", "03_align", _with_pose_source("align", ("reference",)), align_params, run_align,
         poses="aligned.star"),
    Step("msa", "04_msa", ("align",)),
    Step("classify", "05_classify", _with_pose_source("classify", classify_upstream), classify_params, run_classify),
    Step("embed", "06_embed", ("classify",)),
    Step("select", "07_select", _with_pose_source("select", ("classify", "reference")), select_params, run_select,
         poses="selected.star"),
    Step("final", "08_final", _with_pose_source("final"), final_params, run_final),
)
BY_NAME = {s.name: s for s in STEPS}


def get_step(name: str) -> Step:
    if name not in BY_NAME:
        raise KeyError(f"unknown step {name!r} (steps: {', '.join(BY_NAME)})")
    return BY_NAME[name]
