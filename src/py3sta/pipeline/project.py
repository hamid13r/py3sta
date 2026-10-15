"""Project folder: ``py3sta init`` and loading an existing project.

    project/
      project.yaml
      input.star                 # copy of the user's STAR, tomogram / Warp XML paths made absolute
      cycle-000/cycle.yaml
      cycle-000/00_input.star -> ../input.star
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..average import per_tomo_files, tomograms_from_star
from ..io.mrc import read_header
from ..io.particles import ParticleSet, tomo_names_of
from ..io.star import read_star, write_star
from ..utils.add_warp_xml import WARP_XML_COLUMN
from .config import (CYCLE_HEADER, PROJECT_HEADER, ComputeConfig, ConfigError, CtfConfig, CycleConfig,
                     ProjectConfig, WedgeConfig, read_cycle_config, read_project_config, write_yaml)

PROJECT_FILE = "project.yaml"
CYCLE_FILE = "cycle.yaml"
CYCLE_INPUT = "00_input.star"
CYCLE_RE = re.compile(r"^cycle-(\d{3,})$")
PATH_COLUMNS = ("rlnMicrographName", WARP_XML_COLUMN)


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

    def cycle_dir(self, n: int) -> Path:
        return self.root / cycle_dirname(n)

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


def init_project(root, star, box: int | None = None, ctf_mode: str = "none", wedge: bool | None = None,
                 backend: str = "numpy", gpu: int | str = 0, jobs: int = 0, force: bool = False) -> InitReport:
    """Create a project folder from a particle STAR file.

    Checks that every tomogram (rlnMicrographName) and Warp XML (rlnWarpXMLfile) exists and
    that all tomograms share one voxel size, then writes project.yaml, input.star (paths made
    absolute) and cycle-000 with default parameters. ``box`` defaults to rlnImageSize;
    ``wedge`` defaults to True when the STAR file has rlnWarpXMLfile.
    Refuses to overwrite an existing project unless ``force``; ``force`` rewrites only
    project.yaml, input.star and cycle-000/cycle.yaml.
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

    cfg = ProjectConfig(box=box, apix=apix, ts_apix=ps.ts_apix,
                        wedge=WedgeConfig(use=wedge), ctf=CtfConfig(mode=ctf_mode),
                        compute=ComputeConfig(backend=backend, gpu=gpu, jobs=jobs))
    cfg.validate()
    cycle = CycleConfig.default_for(cfg, 0)
    cycle.validate(cfg)

    # write: input.star with absolute paths (keep symlink names: tomo_key() reads the file name)
    root.mkdir(parents=True, exist_ok=True)
    df = blocks[pblock]
    names = tomo_names_of(ps.df)
    for col, files in (("rlnMicrographName", tomos), (WARP_XML_COLUMN, xmls)):
        if col in df and files:
            df[col] = names.map(lambda t: os.path.abspath(files[t])).to_numpy()
    write_star(blocks, root / cfg.input_star)
    write_yaml(cfg, root / PROJECT_FILE, PROJECT_HEADER)
    cdir = root / cycle_dirname(0)
    cdir.mkdir(exist_ok=True)
    write_yaml(cycle, cdir / CYCLE_FILE, CYCLE_HEADER)
    link = cdir / CYCLE_INPUT
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path("..") / cfg.input_star)

    per_tomo = names.value_counts(sort=False).to_dict()
    return InitReport(Project(root, cfg), len(ps), per_tomo, tomos, tilt_ranges, warnings)
