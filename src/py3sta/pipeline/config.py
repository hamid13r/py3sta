"""project.yaml and cycle.yaml: dataclasses, YAML I/O and validation.

Units: physical sizes in Å, box sizes in tomogram voxels, angles in degrees.
Unknown keys are errors (typos must not be silently ignored); every error names
the file and the dotted key, e.g. ``cycle.yaml: reference.mask.radius: must be > 0 Å (got -5)``.
"""
from __future__ import annotations

import dataclasses
import difflib
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SCHEMA = 1

CTF_MODES = ("none", "flip", "wiener", "amplitude")
BACKENDS = ("numpy", "torch")
WEDGE_SIGNS = ("auto", 1, -1)
REFERENCE_MODES = ("global", "file", "classes")
MASK_SHAPES = ("sphere", "ellipsoid", "box", "file")
STACK_DTYPES = ("float16", "float32")
CLASSIFY_METHODS = ("column", "kmeans", "ward", "gmm")
MSA_MODES = ("wmd", "plain")


class ConfigError(ValueError):
    """Invalid configuration; the message names the file and key."""


# ---------------------------------------------------------------- schema
@dataclass
class WedgeConfig:
    use: bool = True
    sign: typing.Union[str, int] = "auto"      # auto (from Warp XML AreAnglesInverted) | 1 | -1
    dose_weight: bool = False


@dataclass
class CtfConfig:
    mode: str = "none"                          # none | flip | wiener | amplitude (see ctf-check)
    height_sign: float = 1.0


@dataclass
class StackSettings:
    margin: int = 4                             # voxels added on each side of the box (drift allowance)
    dtype: str = "float16"                      # float16 | float32


@dataclass
class ComputeConfig:
    backend: str = "numpy"                      # numpy | torch
    gpu: typing.Union[int, str] = 0             # GPU index, or "cpu" for torch on CPU
    jobs: int = 0                               # CPU worker processes (0 = all cores)


@dataclass
class ProjectConfig:
    box: int                                    # voxels at the tomogram pixel size
    apix: float                                 # Å, tomogram voxel size (MRC header)
    ts_apix: float                              # Å, pixel size of rlnCoordinate* (STAR)
    input_star: str = "input.star"              # relative to the project directory
    particle_radius: typing.Optional[float] = None  # Å, normalisation sphere; null = half the box
    invert_contrast: bool = False               # true: densities * -1 on read (WBP tomograms -> protein white)
    wedge: WedgeConfig = field(default_factory=WedgeConfig)
    stack: StackSettings = field(default_factory=StackSettings)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    schema: int = SCHEMA

    def validate(self, where: str = "project.yaml") -> None:
        err = _Errors(where)
        err.check(self.schema == SCHEMA, "schema", f"unsupported schema {self.schema} (this py3sta reads {SCHEMA})")
        err.check(self.box >= 8, "box", f"must be >= 8 voxels (got {self.box})")
        err.check(self.apix > 0, "apix", f"must be > 0 Å (got {self.apix})")
        err.check(self.ts_apix > 0, "ts_apix", f"must be > 0 Å (got {self.ts_apix})")
        r = self.particle_radius
        err.check(r is None or 0 < r <= self.box * self.apix / 2, "particle_radius",
                  f"must be > 0 and <= half the box ({self.box * self.apix / 2:g} Å), or null (got {r})")
        err.choice(self.wedge.sign, WEDGE_SIGNS, "wedge.sign")
        err.check(self.wedge.use or not self.wedge.dose_weight, "wedge.dose_weight", "needs wedge.use: true")
        err.check(self.stack.margin >= 1, "stack.margin", f"must be >= 1 voxel (got {self.stack.margin})")
        err.choice(self.stack.dtype, STACK_DTYPES, "stack.dtype")
        err.choice(self.compute.backend, BACKENDS, "compute.backend")
        g = self.compute.gpu
        err.check(g == "cpu" or (isinstance(g, int) and g >= 0), "compute.gpu",
                  f"must be a GPU index >= 0 or 'cpu' (got {g!r})")
        err.check(self.compute.jobs >= 0, "compute.jobs", f"must be >= 0 (got {self.compute.jobs})")
        err.raise_if_any()

    def align_shift_vox(self, max_shift_A) -> list[int]:
        """Alignment shift limits [x, y, z] in voxels (tomogram axes): max_shift (Å, one value or three) or
        2 voxels, within the stack drift allowance (margin - 2: a fresh crop may already be 0.5 voxel
        off-centre, see stack.max_drift)."""
        lim = max(0, self.stack.margin - 2)
        if max_shift_A is None:
            return [min(lim, 2)] * 3
        v = max_shift_A if isinstance(max_shift_A, list) else [max_shift_A] * 3
        return [min(lim, int(x / self.apix + 1e-6)) for x in v]

    @property
    def box_angst(self) -> float:
        return self.box * self.apix

    @property
    def norm_radius_vox(self) -> float:
        """Normalisation radius in voxels (particle_radius, default half the box)."""
        return (self.particle_radius or self.box_angst / 2) / self.apix


@dataclass
class AverageConfig:
    halves: bool = True
    normalize: bool = True
    cc_lowpass: float = 40.0                    # Å, low-pass for the per-tomogram CC check


@dataclass
class MaskConfig:
    shape: str = "sphere"                       # sphere | ellipsoid | box | file
    radius: typing.List[float] = field(default_factory=lambda: [150.0])  # Å: 1 value (sphere) or 3 (x, y, z)
    soft: float = 20.0                          # Å, cosine edge width
    center: typing.List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # Å from the box centre
    file: typing.Optional[str] = None           # shape: file


@dataclass
class ReferenceConfig:
    mode: str = "global"                        # global | file | classes
    files: typing.List[str] = field(default_factory=list)  # mode: file (MRC volumes or MRC volume stacks)
    invert: bool = False                        # flip the contrast of the reference files
    lowpass: typing.Optional[float] = 30.0      # Å, null = off
    highpass: typing.Optional[float] = None     # Å, null = off
    mask: MaskConfig = field(default_factory=MaskConfig)


@dataclass
class AlignConfig:
    enable: bool = True                         # false: poses passed through unchanged
    levels: typing.List[typing.List[float]] = field(default_factory=lambda: [[12.0, 4.0, 12.0, 4.0], [3.0, 1.0, 3.0, 1.0],
                                                                             [1.0, 0.5, 1.0, 0.5]])
    max_shift: typing.Optional[typing.Union[float, typing.List[float]]] = None  # Å: one value or [x, y, z] along the
                                                # tomogram axes (<= stack.margin - 2 voxels); null = 2 voxels
    lowpass: typing.Optional[float] = 30.0      # Å
    highpass: typing.Optional[float] = None     # Å
    bin: int = 1
    wedge: bool = True                          # constrained CC (needs wedge.use in project.yaml)
    mask: typing.Optional[MaskConfig] = None    # null: the reference mask (02_reference/mask.mrc)
    min_gain: float = 0.0                       # opt-in restraint: keep the pose unless the CC gain is >= this


@dataclass
class MSAConfig:
    enable: bool = True                         # false: no MSA (classify.method column only)
    mode: str = "wmd"                           # wmd: wedge-masked differences | plain: x - mean (i3)
    wedge: bool = True                          # plain: filter each particle with its own wedge first
    mask: typing.Optional[MaskConfig] = None    # null: the reference mask (02_reference/mask.mrc)
    bin: int = 2
    lowpass: typing.Optional[float] = 30.0      # Å
    highpass: typing.Optional[float] = None     # Å
    n_factors: int = 20


@dataclass
class ClassifyConfig:
    method: str = "kmeans"                      # kmeans | ward | gmm (on MSA factors) | column (an integer STAR column)
    column: str = "rlnClassNumber"              # method column; a missing column = one class
    factors: typing.List[int] = field(default_factory=lambda: list(range(1, 9)))  # MSA factors used (i3 CLSFACT)
    k: typing.List[int] = field(default_factory=lambda: [2, 3, 4, 6, 8])          # numbers of classes tried
    use: typing.Optional[int] = None            # the K passed on to select (null: the only K in k)
    standardize: bool = False                   # scale each factor to unit variance first
    n_init: int = 10                            # kmeans / gmm restarts
    bootstrap: int = 20                         # stability: re-clusterings of random 80 % subsets (0 = off)
    seed: int = 0


@dataclass
class SelectConfig:
    align: typing.Union[str, typing.List[int]] = "all"   # classes aligned to the target ("all" = the rest)
    leave: typing.List[int] = field(default_factory=list)    # poses kept
    exclude: typing.List[int] = field(default_factory=list)  # members get py3staExcluded = 1
    target: str = "reference"                   # reference (02_reference, reference 1) | path to an MRC
    lowpass: typing.Optional[float] = 30.0      # Å, alignment search
    highpass: typing.Optional[float] = None     # Å
    bin: int = 2                                # binning of the global search
    step: float = 10.0                          # deg, global angular step
    max_shift: float = 20.0                     # Å


@dataclass
class CycleConfig:
    cycle: int = 0
    ctf: CtfConfig = field(default_factory=CtfConfig)  # belongs to the tomograms in 00_input.star
    average: AverageConfig = field(default_factory=AverageConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    msa: MSAConfig = field(default_factory=MSAConfig)
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    schema: int = SCHEMA

    def validate(self, project: ProjectConfig | None = None, where: str = "cycle.yaml") -> None:
        """Checks values; with ``project``, also checks sizes against the box."""
        err = _Errors(where)
        err.check(self.schema == SCHEMA, "schema", f"unsupported schema {self.schema} (this py3sta reads {SCHEMA})")
        err.check(self.cycle >= 0, "cycle", f"must be >= 0 (got {self.cycle})")
        err.choice(self.ctf.mode, CTF_MODES, "ctf.mode")
        err.check(self.ctf.height_sign in (1, -1), "ctf.height_sign", f"must be 1 or -1 (got {self.ctf.height_sign})")
        if project is not None:
            err.check(self.ctf.mode == "none" or project.wedge.use, "ctf.mode",
                      f"{self.ctf.mode!r} needs tilt information: set wedge.use: true in project.yaml")
        r = self.reference
        err.choice(r.mode, REFERENCE_MODES, "reference.mode")
        err.check(r.mode != "file" or r.files, "reference.files", "mode 'file' needs at least one MRC file")
        err.check(r.mode != "classes", "reference.mode", "'classes' is not implemented yet (comes with select, Phase 5)")
        for key, v in (("lowpass", r.lowpass), ("highpass", r.highpass)):
            err.check(v is None or v > 0, f"reference.{key}", f"must be > 0 Å or null (got {v})")
        if r.lowpass and r.highpass:
            err.check(r.highpass > r.lowpass, "reference.highpass",
                      f"must be larger than lowpass (resolution in Å; got highpass {r.highpass}, lowpass {r.lowpass})")
        m = r.mask
        err.choice(m.shape, MASK_SHAPES, "reference.mask.shape")
        if m.shape == "file":
            err.check(bool(m.file), "reference.mask.file", "shape 'file' needs a mask file")
        else:
            n = 1 if m.shape == "sphere" else 3
            err.check(len(m.radius) == n, "reference.mask.radius",
                      f"shape {m.shape!r} needs {n} value(s) in Å (got {len(m.radius)})")
            err.check(all(v > 0 for v in m.radius), "reference.mask.radius", f"must be > 0 Å (got {m.radius})")
        err.check(m.soft >= 0, "reference.mask.soft", f"must be >= 0 Å (got {m.soft})")
        err.check(len(m.center) == 3, "reference.mask.center", f"needs 3 values in Å (got {len(m.center)})")
        if project is not None:
            if m.shape != "file" and len(m.center) == 3 and m.radius:
                half = project.box_angst / 2
                reach = max(m.radius) + m.soft + max(abs(c) for c in m.center)
                err.check(reach <= half, "reference.mask.radius",
                          f"radius + soft + |center| = {reach:g} Å exceeds half the box ({half:g} Å)")
            if r.lowpass:
                err.check(r.lowpass >= 2 * project.apix, "reference.lowpass",
                          f"{r.lowpass:g} Å is beyond Nyquist ({2 * project.apix:g} Å)")
        a = self.align
        for i, lv in enumerate(a.levels):
            ok = len(lv) == 4 and all(v >= 0 for v in lv) and lv[0] <= 180 and lv[2] <= 180 \
                and (lv[0] == 0 or lv[1] > 0) and (lv[2] == 0 or lv[3] > 0)
            err.check(ok, f"align.levels[{i}]", f"must be [nut, nstep, spin, sstep] (deg, >= 0, steps > 0 when the "
                      f"range is > 0, ranges <= 180; got {lv})")
        for key, v in (("lowpass", a.lowpass), ("highpass", a.highpass)):
            err.check(v is None or v > 0, f"align.{key}", f"must be > 0 Å or null (got {v})")
        err.check(a.bin >= 1, "align.bin", f"must be >= 1 (got {a.bin})")
        ms = [] if a.max_shift is None else (a.max_shift if isinstance(a.max_shift, list) else [a.max_shift])
        err.check(len(ms) in (0, 1, 3) and all(v >= 0 for v in ms), "align.max_shift",
                  f"must be one value or [x, y, z], >= 0 Å, or null (got {a.max_shift})")
        err.check(a.min_gain >= 0, "align.min_gain", f"must be >= 0 (got {a.min_gain})")
        if a.mask is not None:
            err.choice(a.mask.shape, MASK_SHAPES, "align.mask.shape")
        if project is not None:
            lim = (project.stack.margin - 2) * project.apix
            err.check(all(v <= lim + 1e-6 for v in ms), "align.max_shift",
                      f"{a.max_shift} Å exceeds the stack drift allowance ({lim:g} Å = stack.margin - 2 voxels); "
                      "raise stack.margin in project.yaml (rebuilds the stack) or lower align.max_shift")
            err.check(project.box % a.bin == 0, "align.bin", f"box {project.box} is not divisible by {a.bin}")
            if a.lowpass:
                err.check(a.lowpass >= 2 * project.apix * a.bin, "align.lowpass",
                          f"{a.lowpass:g} Å is beyond Nyquist of the binned search ({2 * project.apix * a.bin:g} Å)")
            err.check(not a.wedge or project.wedge.use or not a.enable, "align.wedge",
                      "needs tilt information (wedge.use: true in project.yaml); or set align.wedge: false")
        ms = self.msa
        err.choice(ms.mode, MSA_MODES, "msa.mode")
        for key, v in (("lowpass", ms.lowpass), ("highpass", ms.highpass)):
            err.check(v is None or v > 0, f"msa.{key}", f"must be > 0 Å or null (got {v})")
        err.check(ms.bin >= 1, "msa.bin", f"must be >= 1 (got {ms.bin})")
        err.check(ms.n_factors >= 1, "msa.n_factors", f"must be >= 1 (got {ms.n_factors})")
        if ms.mask is not None:
            err.choice(ms.mask.shape, MASK_SHAPES, "msa.mask.shape")
        if project is not None and ms.enable:
            err.check(project.box % ms.bin == 0, "msa.bin", f"box {project.box} is not divisible by {ms.bin}")
            if ms.lowpass:
                err.check(ms.lowpass >= 2 * project.apix * ms.bin, "msa.lowpass",
                          f"{ms.lowpass:g} Å is beyond Nyquist of the binned MSA stack ({2 * project.apix * ms.bin:g} Å)")
            uses_wedge = ms.mode == "wmd" or ms.wedge
            err.check(not uses_wedge or project.wedge.use, "msa.mode",
                      f"{ms.mode}{' + wedge' if ms.mode == 'plain' else ''} needs tilt information (wedge.use: true in "
                      "project.yaml); or use mode: plain with wedge: false")
        cl = self.classify
        err.choice(cl.method, CLASSIFY_METHODS, "classify.method")
        if cl.method != "column":
            err.check(ms.enable, "classify.method", f"{cl.method!r} classifies MSA factors: set msa.enable: true "
                      "(or use method: column)")
            err.check(bool(cl.factors) and len(set(cl.factors)) == len(cl.factors)
                      and all(1 <= f <= ms.n_factors for f in cl.factors), "classify.factors",
                      f"must be distinct factor numbers in 1..msa.n_factors ({ms.n_factors}) (got {cl.factors})")
            err.check(bool(cl.k) and all(k >= 2 for k in cl.k) and len(set(cl.k)) == len(cl.k), "classify.k",
                      f"must be distinct numbers of classes >= 2 (got {cl.k})")
            err.check(cl.use is None or cl.use in cl.k, "classify.use", f"must be one of classify.k {cl.k} (got {cl.use})")
            err.check(cl.n_init >= 1, "classify.n_init", f"must be >= 1 (got {cl.n_init})")
            err.check(cl.bootstrap >= 0, "classify.bootstrap", f"must be >= 0 (got {cl.bootstrap})")
        sel = self.select
        err.check(sel.align == "all" or isinstance(sel.align, list), "select.align",
                  f"must be 'all' or a list of class numbers (got {sel.align!r})")
        lists = {"align": sel.align if isinstance(sel.align, list) else [], "leave": sel.leave, "exclude": sel.exclude}
        for a, b in (("align", "leave"), ("align", "exclude"), ("leave", "exclude")):
            both = sorted(set(lists[a]) & set(lists[b]))
            err.check(not both, f"select.{b}", f"class(es) {both} also in select.{a}")
        for key, v in (("lowpass", sel.lowpass), ("highpass", sel.highpass)):
            err.check(v is None or v > 0, f"select.{key}", f"must be > 0 Å or null (got {v})")
        err.check(sel.bin >= 1, "select.bin", f"must be >= 1 (got {sel.bin})")
        err.check(0 < sel.step <= 90, "select.step", f"must be in (0, 90] degrees (got {sel.step})")
        err.check(sel.max_shift >= 0, "select.max_shift", f"must be >= 0 Å (got {sel.max_shift})")
        if project is not None and sel.lowpass:
            err.check(sel.lowpass >= 2 * project.apix * sel.bin, "select.lowpass",
                      f"{sel.lowpass:g} Å is beyond Nyquist of the binned search ({2 * project.apix * sel.bin:g} Å); "
                      "lower select.bin or raise select.lowpass")
        err.raise_if_any()

    @classmethod
    def default_for(cls, project: ProjectConfig, cycle: int = 0) -> "CycleConfig":
        """Defaults scaled to the box: mask radius 40 % and soft edge 10 % of the box."""
        c = cls(cycle=cycle)
        box = project.box_angst
        c.reference.mask.radius = [_round(0.4 * box)]
        c.reference.mask.soft = _round(0.1 * box)
        c.reference.lowpass = max(30.0, _round(4 * project.apix))
        c.select.lowpass = c.reference.lowpass
        c.align.lowpass = c.reference.lowpass
        c.align.wedge = project.wedge.use
        c.msa.lowpass = c.reference.lowpass
        if not project.wedge.use:
            c.msa.mode, c.msa.wedge = "plain", False
        c.select.max_shift = _round(0.1 * box)
        return c


def _round(v: float) -> float:
    return float(round(v, 1))


# ----------------------------------------------------------------- YAML I/O
PROJECT_HEADER = """\
# py3sta project configuration
# box: voxels at the tomogram pixel size; apix / ts_apix: Å (tomogram voxel / STAR coordinates)
# particle_radius: Å, sphere for per-particle normalisation (null = half the box)
# invert_contrast: true multiplies densities by -1 when particles are read (WBP tomograms: protein becomes white)
# wedge.sign: auto | 1 | -1;  stack.margin: voxels per side (drift allowance);  stack.dtype: float16 | float32
# compute.backend: numpy | torch;  compute.gpu: index or 'cpu';  compute.jobs: 0 = all cores
"""

CYCLE_HEADER = """\
# py3sta cycle configuration (copied to the next cycle by 'py3sta next', then edit)
# ctf.mode: none | flip | wiener | amplitude -- belongs to the tomograms in 00_input.star ('py3sta ctf-check')
# All sizes in Å.  reference.mode: global | file | classes;  mask.shape: sphere | ellipsoid | box | file
# mask.radius: 1 value (sphere) or 3 (ellipsoid / box half-sizes); reference.invert flips reference files only
# align: levels [[nut, nstep, spin, sstep], ...] coarse to fine around the current pose (deg); max_shift Å per axis
#   or [x, y, z] along the tomogram axes (<= stack.margin - 2 voxels; z is the noisiest: missing wedge);
#   mask: null = reference mask; min_gain: opt-in, keep a particle's pose unless the CC improves by >= min_gain
# msa: mode wmd (wedge-masked differences: no missing-wedge factors) | plain (x - mean, i3; wedge: true filters
#   each particle with its own wedge); mask: null = reference mask; bin, lowpass / highpass Å; n_factors
# classify.method: kmeans | ward | gmm on the MSA factors listed in classify.factors, one classification per K in
#   classify.k; use: the K passed to select (null = the only K) | column (classes from classify.column)
# select: per class align | leave | exclude ("align: all" = every class not listed elsewhere);
#   target: reference (02_reference, reference 1) or an MRC path; lowpass / highpass / max_shift in Å, step in deg
"""


def to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)


def write_yaml(cfg, path, header: str = "") -> None:
    text = yaml.safe_dump(to_dict(cfg), sort_keys=False, default_flow_style=None)
    Path(path).write_text(header + text)


def read_project_config(path) -> ProjectConfig:
    cfg = _load(ProjectConfig, path)
    cfg.validate(where=str(path))
    return cfg


def read_cycle_config(path, project: ProjectConfig | None = None) -> CycleConfig:
    cfg = _load(CycleConfig, path)
    cfg.validate(project, where=str(path))
    return cfg


def _load(cls, path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"{path}: file not found")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML: {e}") from None
    return from_dict(cls, data or {}, str(path))


def from_dict(cls, data, where: str, prefix: str = ""):
    """Build dataclass ``cls`` from a dict: unknown keys, missing required keys and wrong types are errors."""
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: {prefix.rstrip('.') or 'top level'}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = [k for k in data if k not in fields]
    if unknown:
        k = unknown[0]
        close = difflib.get_close_matches(str(k), fields, n=1)
        hint = f" (did you mean {prefix}{close[0]!s}?)" if close else f" (allowed: {', '.join(fields)})"
        raise ConfigError(f"{where}: {prefix}{k}: unknown key{hint}")
    kw = {}
    for name, f in fields.items():
        key = prefix + name
        if name not in data:
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                raise ConfigError(f"{where}: {key}: required")
            continue
        kw[name] = _convert(hints[name], data[name], where, key)
    return cls(**kw)


def _convert(tp, v, where, key):
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, v if v is not None else {}, where, key + ".")
    if origin in (typing.Union, getattr(types, "UnionType", typing.Union)):
        if v is None and type(None) in args:
            return None
        for a in args:
            if a is type(None):
                continue
            try:
                return _convert(a, v, where, key)
            except ConfigError:
                pass
        names = " or ".join("null" if a is type(None) else getattr(a, "__name__", str(a)) for a in args)
        raise ConfigError(f"{where}: {key}: expected {names}, got {v!r}")
    if origin in (list, typing.List):
        items = v if isinstance(v, list) else [v]          # a scalar is accepted as a 1-item list
        return [_convert(args[0], x, where, f"{key}[{i}]") for i, x in enumerate(items)]
    if tp is bool:
        if isinstance(v, bool):
            return v
    elif tp is int:
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    elif tp is float:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    elif tp is str:
        if isinstance(v, str):
            return v
    raise ConfigError(f"{where}: {key}: expected {tp.__name__}, got {v!r}")


class _Errors:
    """Collects all validation errors so the user sees them in one go."""

    def __init__(self, where: str):
        self.where, self.msgs = where, []

    def check(self, ok, key: str, msg: str) -> None:
        if not ok:
            self.msgs.append(f"{key}: {msg}")

    def choice(self, v, allowed, key: str) -> None:
        self.check(v in allowed, key, f"must be one of {', '.join(map(str, allowed))} (got {v!r})")

    def raise_if_any(self) -> None:
        if self.msgs:
            raise ConfigError("\n".join(f"{self.where}: {m}" for m in self.msgs))
