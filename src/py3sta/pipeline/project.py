"""Project folder: ``py3sta init`` and loading an existing project.

    project/
      project.yaml
      input.star                 # copy of the user's STAR, tomogram / Warp XML paths made absolute
      stack/                     # raw particle crops shared by all cycles (stack.py)
      cycle-000/cycle.yaml
      cycle-000/00_input.star    # copy of input.star (editable)
"""
from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..average import per_tomo_files, tomograms_from_star
from ..io.mrc import read_header
from ..io.particles import ParticleSet, tomo_names_of
from ..io.star import read_star, write_star
from ..utils.add_warp_xml import WARP_XML_COLUMN
from .config import (CYCLE_HEADER, PROJECT_HEADER, ComputeConfig, ConfigError, CycleConfig, ProjectConfig,
                     WedgeConfig, read_cycle_config, read_project_config, write_yaml)

PROJECT_FILE = "project.yaml"
CYCLE_FILE = "cycle.yaml"
CYCLE_INPUT = "00_input.star"
CYCLE_RE = re.compile(r"^cycle-(\d{3,})$")
STACK_DIR = "stack"
PARTICLE_NAME = "rlnTomoParticleName"
EXCLUDED = "py3staExcluded"


def cycle_dirname(n: int) -> str:
    return f"cycle-{n:03d}"


@dataclass
class Project:
    root: Path
    config: ProjectConfig

    @property
    def input_star(self) -> Path:
        return self.root / self.config.input_star

    @property
    def cycles(self) -> list[int]:
        found = (CYCLE_RE.match(p.name) for p in self.root.iterdir() if p.is_dir())
        return sorted(int(m.group(1)) for m in found if m)

    @property
    def stack_dir(self) -> Path:
        return self.root / STACK_DIR

    def cycle_dir(self, n: int) -> Path:
        return self.root / cycle_dirname(n)

    def cycle_input(self, n: int) -> Path:
        return self.cycle_dir(n) / CYCLE_INPUT

    def cycle_config(self, n: int) -> CycleConfig:
        path = self.cycle_dir(n) / CYCLE_FILE
        cfg = read_cycle_config(path, self.config)
        if cfg.cycle != n:
            raise ConfigError(f"{path}: cycle: {cfg.cycle} does not match the folder {cycle_dirname(n)}")
        return cfg


@dataclass
class InitReport:
    project: Project
    n_particles: int
    per_tomo: dict[str, int]
    tomograms: dict[str, Path]
    tilt_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def load_project(root) -> Project:
    root = Path(root)
    cfg = read_project_config(root / PROJECT_FILE)
    return Project(root, cfg)


@dataclass
class CycleInput:
    """A cycle's particles (excluded rows removed) with their tomograms and tilt information."""
    particles: ParticleSet           # active particles only
    n_total: int
    n_excluded: int
    tomograms: dict                  # {rlnTomoName: Path}
    tilt_files: dict                 # {rlnTomoName: Path} (Warp XML), empty without wedge


def load_particles(project: Project, star) -> tuple[ParticleSet, dict, dict]:
    """Read and check a particle STAR file against the project (also after hand edits): pixel size,
    unique rlnTomoParticleName, all tomograms present with the project's voxel size, Warp XMLs present
    if the wedge is used. Returns (all particles, tomograms, Warp XMLs)."""
    cfg = project.config
    try:
        ps = ParticleSet.from_star(star)
    except (ValueError, NotImplementedError) as e:
        raise ConfigError(f"{star}: {e}") from None
    if abs(ps.ts_apix - cfg.ts_apix) > 1e-4:
        raise ConfigError(f"{star}: coordinate pixel size {ps.ts_apix:g} Å differs from the project's {cfg.ts_apix:g} Å")
    if PARTICLE_NAME not in ps.df:
        raise ConfigError(f"{star}: no {PARTICLE_NAME} column (needed to match particles with the stack)")
    names = ps.df[PARTICLE_NAME].astype(str)
    if names.duplicated().any():
        raise ConfigError(f"{star}: {PARTICLE_NAME} not unique: " + ", ".join(names[names.duplicated()].unique()[:5]))
    tomos = tomograms_from_star(ps, star)
    odd = {t: a for t, a in ((t, round(read_header(p)["apix"], 4)) for t, p in tomos.items())
           if abs(a - cfg.apix) > 1e-3}
    if odd:
        raise ConfigError(f"{star}: tomogram voxel size differs from the project's {cfg.apix:g} Å: "
                          + ", ".join(f"{t} {a:g} Å" for t, a in list(odd.items())[:5]))
    xmls = per_tomo_files(ps, star, WARP_XML_COLUMN) if cfg.wedge.use else {}
    return ps, tomos, xmls


def load_cycle_input(project: Project, n: int) -> CycleInput:
    path = project.cycle_input(n)
    if not path.is_file():
        raise ConfigError(f"{path}: missing")
    ps, tomos, xmls = load_particles(project, path)
    n_total = len(ps)
    if EXCLUDED in ps.df:
        keep = ps.df[EXCLUDED].fillna(0).astype(int) != 1
        ps.df = ps.df[keep].reset_index(drop=True)
    active = set(ps.df["rlnTomoName"])
    return CycleInput(ps, n_total, n_total - len(ps), {t: p for t, p in tomos.items() if t in active},
                      {t: p for t, p in xmls.items() if t in active})


def init_project(root, star, box: int | None = None, ctf_mode: str = "none", wedge: bool | None = None,
                 backend: str = "numpy", gpu: int | str = 0, jobs: int = 0, force: bool = False,
                 particle_radius: float | None = None, margin: int = 4) -> InitReport:
    """Create a project folder from a particle STAR file.

    Checks that every tomogram (rlnMicrographName) and Warp XML (rlnWarpXMLfile) exists and
    that all tomograms share one voxel size, then writes project.yaml, input.star (paths made
    absolute, rlnTomoParticleName added if missing) and cycle-000 (cycle.yaml with ``ctf_mode``,
    00_input.star = copy of input.star). ``box`` defaults to rlnImageSize;
    ``wedge`` defaults to True when the STAR file has rlnWarpXMLfile.
    Refuses to overwrite an existing project unless ``force``; ``force`` rewrites only
    project.yaml, input.star and cycle-000/{cycle.yaml, 00_input.star}.
    """
    from ..wedge import TiltInfo

    root, star = Path(root), Path(star)
    if (root / PROJECT_FILE).exists() and not force:
        raise ConfigError(f"{root / PROJECT_FILE} exists (use --force to overwrite)")
    if not star.is_file():
        raise ConfigError(f"{star}: STAR file not found")

    blocks = read_star(star)
    pblock = "particles" if "particles" in blocks else next(iter(blocks))
    ps = ParticleSet.from_star(star)
    warnings = []

    # tomograms: all present, one voxel size
    tomos = tomograms_from_star(ps, star)
    apixes = {t: round(read_header(p)["apix"], 4) for t, p in tomos.items()}   # MRC header is float32
    apix = next(iter(apixes.values()))
    odd = {t: a for t, a in apixes.items() if abs(a - apix) > 1e-3}
    if odd:
        raise ConfigError(f"tomograms differ in voxel size: {apix:g} Å for most, but "
                          + ", ".join(f"{t} {a:g} Å" for t, a in list(odd.items())[:5]))

    if box is None:
        optics = blocks.get("optics")
        if optics is None or "rlnImageSize" not in optics:
            raise ConfigError(f"{star}: no rlnImageSize in data_optics; give the box size")
        box = int(optics["rlnImageSize"].iloc[0])

    # Warp XML: present for all tomograms if the column exists
    xmls, tilt_ranges = {}, {}
    if WARP_XML_COLUMN in ps.df:
        xmls = per_tomo_files(ps, star, WARP_XML_COLUMN)
        for t, p in xmls.items():
            ti = TiltInfo.from_warp_xml(p)
            tilt_ranges[t] = ti.range
            if ctf_mode != "none" and ti.ctf is None:
                raise ConfigError(f"ctf mode {ctf_mode!r}: no CTF information in {p}")
    if wedge is None:
        wedge = bool(xmls)
    if wedge and not xmls:
        raise ConfigError(f"{star}: no {WARP_XML_COLUMN} column; add it with 'py3sta add-xml' or disable the wedge")
    if not wedge:
        warnings.append("no tilt information: averages will not be wedge-weighted")
    if "rlnRandomSubset" not in ps.df:
        warnings.append("no rlnRandomSubset: half sets will alternate rows")

    cfg = ProjectConfig(box=box, apix=apix, ts_apix=ps.ts_apix, particle_radius=particle_radius,
                        wedge=WedgeConfig(use=wedge), compute=ComputeConfig(backend=backend, gpu=gpu, jobs=jobs))
    cfg.stack.margin = margin
    cfg.validate()
    cycle = CycleConfig.default_for(cfg, 0)
    cycle.ctf.mode = ctf_mode
    cycle.validate(cfg)

    # write: input.star with absolute paths (keep symlink names: tomo_key() reads the file name)
    root.mkdir(parents=True, exist_ok=True)
    df = blocks[pblock]
    names = tomo_names_of(ps.df)
    for col, files in (("rlnMicrographName", tomos), (WARP_XML_COLUMN, xmls)):
        if col in df and files:
            df[col] = names.map(lambda t: os.path.abspath(files[t])).to_numpy()
    if PARTICLE_NAME not in df:
        df[PARTICLE_NAME] = (names + "/" + (names.groupby(names).cumcount() + 1).astype(str)).to_numpy()
    elif df[PARTICLE_NAME].astype(str).duplicated().any():
        raise ConfigError(f"{star}: {PARTICLE_NAME} is not unique")
    write_star(blocks, root / cfg.input_star)
    write_yaml(cfg, root / PROJECT_FILE, PROJECT_HEADER)
    cdir = root / cycle_dirname(0)
    cdir.mkdir(exist_ok=True)
    write_yaml(cycle, cdir / CYCLE_FILE, CYCLE_HEADER)
    dst = cdir / CYCLE_INPUT
    if dst.is_symlink():
        dst.unlink()
    shutil.copyfile(root / cfg.input_star, dst)

    per_tomo = names.value_counts(sort=False).to_dict()
    return InitReport(Project(root, cfg), len(ps), per_tomo, tomos, tilt_ranges, warnings)


@dataclass
class NextReport:
    n: int                           # the new cycle
    n_particles: int
    source: Path                     # the final STAR copied to 00_input.star
    warnings: list[str] = field(default_factory=list)


def next_cycle(project: Project, n: int, force: bool = False) -> NextReport:
    """Create cycle n+1: cycle.yaml copied from cycle n (only ``cycle:`` changed, comments kept) and
    00_input.star = copy of cycle n's final STAR (hand edits included).

    Refuses when the final step of cycle n is not done (missing / stale / failed) or cycle n+1
    already exists; ``force`` overrides both (an existing cycle n+1 gets new cycle.yaml / 00_input.star,
    so its steps become stale)."""
    from .runner import step_state
    from .steps import final_name, get_step

    st = step_state(project, n, "final")
    src = project.cycle_dir(n) / get_step("final").folder / final_name(n)
    warnings = []
    if not st.ok:
        if not force or not src.is_file():
            raise ConfigError(f"{cycle_dirname(n)}: final step is {st.describe()}; run "
                              f"'py3sta run {project.root} {n} --from final'"
                              + (" (or use --force)" if src.is_file() else ""))
        warnings.append(f"final step is {st.state}; copying {src.name} anyway (--force)")
    if st.edited:
        warnings.append(f"{', '.join(st.edited)} was edited by hand; the edited version is used")
    dst = project.cycle_dir(n + 1)
    if dst.exists() and not force:
        raise ConfigError(f"{dst} exists (use --force to replace its cycle.yaml and 00_input.star)")
    text = (project.cycle_dir(n) / CYCLE_FILE).read_text()
    text, k = re.subn(r"(?m)^cycle:\s*\d+\s*$", f"cycle: {n + 1}", text, count=1)
    if k != 1:
        raise ConfigError(f"{project.cycle_dir(n) / CYCLE_FILE}: no 'cycle:' line")
    dst.mkdir(exist_ok=True)
    (dst / CYCLE_FILE).write_text(text)
    shutil.copyfile(src, dst / CYCLE_INPUT)
    project.cycle_config(n + 1)                                  # validates the copy
    n_part = len(load_particles(project, dst / CYCLE_INPUT)[0])
    return NextReport(n + 1, n_part, src, warnings)
