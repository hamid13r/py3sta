"""``py3sta flag``: include / exclude particles of a cycle by editing its 00_input.star (PLAN.md §2,
"Excluded particles"). The edit is tracked like any hand edit (the cycle's steps become stale) and
recorded in ``cycle-NNN/flag.log``.

Selections (a particle is selected if it matches any of them):
    class:CYCLE:K[,K..]     rlnClassNumber in that cycle's final STAR (else its 07_select/selected.star)
    tomo:NAME[,NAME..]      tomogram (rlnTomoName, or the name derived from rlnMicrographName)
    excluded:CYCLE          particles excluded in that cycle (py3staExcludedCycle)
    file:PATH               STAR file with rlnTomoParticleName, or a text file with one name per line
    all
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..io.particles import tomo_names_of
from ..io.star import read_star, write_star
from .config import ConfigError
from .manifest import check_state
from .project import EXCLUDED, PARTICLE_NAME, Project, cycle_dirname

EXCLUDED_CYCLE = "py3staExcludedCycle"
FLAG_LOG = "flag.log"
KINDS = ("class", "tomo", "excluded", "file", "all")


@dataclass
class FlagReport:
    n_particles: int
    excluded_before: int
    excluded_after: int
    newly_excluded: int
    newly_included: int
    unchanged: int                    # selected but already in the requested state
    stale_steps: list = field(default_factory=list)   # steps of the cycle that had run (now stale)
    written: bool = False


def _cycle_arg(text: str, spec: str) -> int:
    m = re.fullmatch(r"(?:cycle-)?(\d+)", text)
    if not m:
        raise ConfigError(f"{spec!r}: {text!r} is not a cycle (0, cycle-000)")
    return int(m.group(1))


def _particle_block(blocks: dict) -> pd.DataFrame:
    return blocks["particles"] if "particles" in blocks else next(iter(blocks.values()))


def class_star(project: Project, k: int) -> Path:
    """The STAR with cycle k's class numbers: its final STAR, else its select output."""
    from .steps import final_name, get_step

    d = project.cycle_dir(k)
    for p in (d / get_step("final").folder / final_name(k), d / get_step("select").folder / "selected.star"):
        if p.is_file():
            return p
    raise ConfigError(f"{cycle_dirname(k)}: no final or selected STAR with class numbers (run select first)")


def _ints(text: str, spec: str) -> list[int]:
    try:
        return [int(v) for v in text.split(",") if v != ""]
    except ValueError:
        raise ConfigError(f"{spec!r}: class numbers must be integers") from None


def select_particles(project: Project, df: pd.DataFrame, spec: str) -> np.ndarray:
    """Boolean mask over ``df`` (a particle table with rlnTomoParticleName) for one selection."""
    kind, _, arg = spec.partition(":")
    names = df[PARTICLE_NAME].astype(str)
    if kind == "all" and not arg:
        return np.ones(len(df), bool)
    if kind not in KINDS or not arg:
        raise ConfigError(f"{spec!r}: selection must be one of class:CYCLE:K, tomo:NAME, excluded:CYCLE, file:PATH, all")
    if kind == "class":
        cyc, _, ks = arg.partition(":")
        k = _cycle_arg(cyc, spec)
        wanted = _ints(ks, spec)
        if not wanted:
            raise ConfigError(f"{spec!r}: give class numbers (class:CYCLE:K[,K..])")
        src = _particle_block(read_star(class_star(project, k)))
        if "rlnClassNumber" not in src:
            raise ConfigError(f"{spec!r}: no rlnClassNumber in {class_star(project, k)}")
        cls = pd.Series(src["rlnClassNumber"].to_numpy(), index=src[PARTICLE_NAME].astype(str))
        unknown = sorted(set(wanted) - set(cls.astype(int)))
        if unknown:
            raise ConfigError(f"{spec!r}: no class {unknown} in {cycle_dirname(k)} "
                              f"(classes: {', '.join(map(str, sorted(set(cls.astype(int)))))})")
        return names.map(cls).isin(wanted).to_numpy()
    if kind == "tomo":
        tomos = tomo_names_of(df).astype(str)
        wanted = [t for t in arg.split(",") if t]
        unknown = [t for t in wanted if t not in set(tomos)]
        if unknown:
            raise ConfigError(f"{spec!r}: unknown tomogram(s) {', '.join(unknown[:5])}")
        return tomos.isin(wanted).to_numpy()
    if kind == "excluded":
        k = _cycle_arg(arg, spec)
        if k not in project.cycles:
            raise ConfigError(f"{spec!r}: no {cycle_dirname(k)}")
        if EXCLUDED_CYCLE not in df or EXCLUDED not in df:
            return np.zeros(len(df), bool)
        return ((df[EXCLUDED].fillna(0).astype(int) == 1) & (df[EXCLUDED_CYCLE].fillna(-1).astype(int) == k)).to_numpy()
    # file
    path = Path(arg)
    if not path.is_file():
        raise ConfigError(f"{spec!r}: file not found")
    try:
        blocks = read_star(path)
        block = next((b for b in blocks.values() if PARTICLE_NAME in b), None)
    except Exception:
        block = None
    if block is not None:
        wanted = block[PARTICLE_NAME].astype(str)
    else:
        wanted = pd.Series([s.strip() for s in path.read_text().splitlines() if s.strip() and not s.startswith("#")])
    unknown = sorted(set(wanted) - set(names))
    if unknown:
        raise ConfigError(f"{spec!r}: {len(unknown)} particle name(s) not in this cycle: {', '.join(unknown[:5])}")
    return names.isin(set(wanted)).to_numpy()


def flag_particles(project: Project, n: int, exclude=(), include=(), dry_run: bool = False,
                   command: str | None = None) -> FlagReport:
    """Set py3staExcluded = 1 (``exclude``) / 0 (``include``) in cycle n's 00_input.star.

    exclude also sets py3staExcludedCycle = n; include keeps it (history). A particle selected by both
    is an error. Refuses while a step of the cycle is running."""
    from .steps import STEPS

    if not exclude and not include:
        raise ConfigError("nothing to do: give --exclude and/or --include")
    path = project.cycle_input(n)
    if not path.is_file():
        raise ConfigError(f"{path}: missing")
    states = {s.name: check_state(project.root, project.cycle_dir(n) / s.folder, None) for s in STEPS if s.implemented}
    running = [k for k, st in states.items() if st.state == "running"]
    if running:
        raise ConfigError(f"{cycle_dirname(n)}: step(s) {', '.join(running)} running; flag after they finish")

    blocks = read_star(path)
    df = _particle_block(blocks)
    if PARTICLE_NAME not in df:
        raise ConfigError(f"{path}: no {PARTICLE_NAME} column")
    ex = np.zeros(len(df), bool)
    inc = np.zeros(len(df), bool)
    for spec in exclude:
        ex |= select_particles(project, df, spec)
    for spec in include:
        inc |= select_particles(project, df, spec)
    both = ex & inc
    if both.any():
        raise ConfigError(f"{int(both.sum())} particle(s) selected by both --exclude and --include, e.g. "
                          + ", ".join(df.loc[both, PARTICLE_NAME].astype(str).head(3)))

    cur = (df[EXCLUDED].fillna(0).astype(int).to_numpy() == 1) if EXCLUDED in df else np.zeros(len(df), bool)
    new = cur.copy()
    new[ex] = True
    new[inc] = False
    rep = FlagReport(len(df), int(cur.sum()), int(new.sum()), int((new & ~cur).sum()), int((cur & ~new).sum()),
                     int(((ex | inc) & (new == cur)).sum()),
                     [k for k, st in states.items() if st.state != "missing"])
    if dry_run or (new == cur).all():
        return rep
    df[EXCLUDED] = new.astype(int)
    if EXCLUDED_CYCLE not in df:
        df[EXCLUDED_CYCLE] = -1
    df.loc[new & ~cur, EXCLUDED_CYCLE] = n
    write_star(blocks, path)
    rep.written = True
    line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{command or ''}\texclude={list(exclude)} include={list(include)}"
            f"\t+{rep.newly_excluded} excluded, +{rep.newly_included} included; "
            f"excluded {rep.excluded_before} -> {rep.excluded_after} of {rep.n_particles}\n")
    with open(project.cycle_dir(n) / FLAG_LOG, "a") as f:
        f.write(line)
    return rep
