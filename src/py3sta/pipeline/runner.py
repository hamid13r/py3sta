"""Run cycle steps and report project status (the CLI and the dashboard call these)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .manifest import StepError, StepState, check_state, run_step
from .project import Project, load_cycle_input
from .steps import STEPS, get_step, resolve_device

STACK_REPORT = "refresh.json"


def step_state(project: Project, n: int, name: str, _cache=None) -> StepState:
    """State of one step of cycle n, with upstream staleness propagated."""
    cache = {} if _cache is None else _cache
    if name in cache:
        return cache[name]
    step = get_step(name)
    folder = project.cycle_dir(n) / step.folder
    if not step.implemented:
        st = StepState("not implemented")
    else:
        try:
            params = step.params(project.config, project.cycle_config(n))
        except Exception as e:                     # invalid cycle.yaml: report, do not crash status
            st = check_state(project.root, folder, None)
            st.reasons.append(f"cycle.yaml: {e}")
            cache[name] = st
            return st
        st = check_state(project.root, folder, params, lambda up: step_state(project, n, up, cache))
    cache[name] = st
    return st


def refresh_project_stack(project: Project, n: int, log=None, shift_limit: int = 0):
    """Bring the project stack up to date with cycle n's particles (excluded ones are skipped)."""
    from ..stack import refresh_stack

    cfg = project.config
    cin = load_cycle_input(project, n)
    log = log or (lambda m: None)
    rep = refresh_stack(project.stack_dir, cin.particles, cin.tomograms, cfg.box, cfg.stack.margin, cfg.apix,
                        cfg.stack.dtype, jobs=cfg.compute.jobs or os.cpu_count(), shift_limit=shift_limit)
    if rep.changed:
        what = ("built" if rep.rebuilt else "updated")
        log(f"stack {what}: {rep.n_new} new, {rep.n_tomogram} re-cropped (tomogram changed"
            + (f": {', '.join(rep.changed_tomograms[:3])}" if rep.changed_tomograms else "")
            + f"), {rep.n_drift} re-cropped (drift); {rep.n_rows} rows, {rep.seconds:.1f} s")
        (project.stack_dir / STACK_REPORT).write_text(json.dumps(
            {**rep.__dict__, "cycle": n, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    else:
        log(f"stack up to date ({rep.n_rows} rows)")
    return rep


def run_cycle(project: Project, n: int, start: str | None = None, stop: str | None = None,
              force: bool = False, log=None) -> list[dict]:
    """Refresh the stack, then run the implemented steps of cycle n from ``start`` to ``stop``.

    Refuses when an upstream step outside the range is stale / missing / failed (``force`` overrides),
    or when a step's STAR output was edited (``force`` keeps a backup and re-runs).
    """
    log = log or (lambda m: None)
    names = [s.name for s in STEPS]
    i0 = names.index(get_step(start).name) if start else 0
    i1 = names.index(get_step(stop).name) if stop else len(names) - 1
    todo = [s for s in STEPS[i0:i1 + 1] if s.implemented]
    if not todo:
        raise StepError(f"no implemented step between {names[i0]} and {names[i1]}")
    cycle_cfg = project.cycle_config(n)                    # validates cycle.yaml
    if not force:
        cache = {}
        for s in todo:
            for up in s.upstream:
                if up in {t.name for t in todo}:
                    continue
                us = step_state(project, n, up, cache)
                if not us.ok:
                    raise StepError(f"upstream step {up} is {us.describe()}; run with --from {up} (or --force)")
    refresh_project_stack(project, n, log)
    _, device = resolve_device(project.config.compute)
    out = []
    for s in todo:
        log(f"cycle-{n:03d} {s.folder}: running on {device}")
        folder = project.cycle_dir(n) / s.folder

        def fn(ctx, s=s):
            for up in s.upstream:
                m = step_state(project, n, up).manifest
                if m:
                    ctx.use_step(up, m["run_id"])
            s.run(ctx, project, n, cycle_cfg)

        m = run_step(project.root, folder, s.name, s.params(project.config, cycle_cfg), fn, force=force,
                     device=device, log=log)
        log(f"cycle-{n:03d} {s.folder}: done in {m['seconds']:.1f} s")
        out.append(m)
    return out


@dataclass
class CycleStatus:
    n: int
    n_particles: int | None
    n_excluded: int | None
    ctf_mode: str | None
    steps: list = field(default_factory=list)      # [(Step, StepState)]
    error: str | None = None


@dataclass
class ProjectStatus:
    cycles: list
    stack_rows: int = 0
    stack_bytes: int = 0
    stack_report: dict | None = None
    drift_warning: int = 0                         # particles of the last cycle beyond half the drift allowance


def project_status(project: Project) -> ProjectStatus:
    from ..stack import ParticleStack, StackError, max_drift, particle_centres

    cycles = []
    for n in project.cycles:
        try:
            cfg = project.cycle_config(n)
            cin = load_cycle_input(project, n)
            cs = CycleStatus(n, cin.n_total, cin.n_excluded, cfg.ctf.mode)
        except Exception as e:
            cs = CycleStatus(n, None, None, None, error=str(e))
        cache = {}
        cs.steps = [(s, step_state(project, n, s.name, cache)) for s in STEPS]
        cycles.append(cs)
    ps = ProjectStatus(cycles)
    try:
        st = ParticleStack.open(project.stack_dir)
        ps.stack_rows, ps.stack_bytes = len(st.table), st.nbytes
        rp = project.stack_dir / STACK_REPORT
        ps.stack_report = json.loads(rp.read_text()) if rp.is_file() else None
        if project.cycles and cycles[-1].error is None:
            cin = load_cycle_input(project, project.cycles[-1])
            names = cin.particles.df["rlnTomoParticleName"].astype(str)
            tab = st.table.set_index(st.table["rlnTomoParticleName"].astype(str))
            known = names.isin(tab.index).to_numpy()
            lo = tab.loc[names[known], ["py3staCropX", "py3staCropY", "py3staCropZ"]].to_numpy(np.int64)
            c = particle_centres(cin.particles, st.apix)[known]
            drift = np.abs(c - (lo + st.size // 2)).max(1)
            ps.drift_warning = int((drift > max_drift(st.margin) / 2).sum())
    except StackError:
        pass
    return ps
