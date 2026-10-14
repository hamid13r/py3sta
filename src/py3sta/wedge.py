"""Missing-wedge (tilt-coverage) model.

Tilt axis is the tomogram Y axis (Warp, IMOD, i3 reconstructions). A tilt image at
angle theta samples the central Fourier plane containing ky and the direction
(cos b, 0, sin b) with b = sign * theta, so a Fourier point k (tomogram frame) is
sampled when its elevation in the kx-kz plane, b = atan(kz/kx), lies inside the
(signed) tilt range.

``sign`` is a property of the reconstruction, not of the particles. For Warp
tomograms with ``AreAnglesInverted="True"`` it is -1: measured on
MIM019_2_lam11_ts_003 (tilts -54.6..+67.6) the power spectrum extends to
b = -67 and only to b = +55. Use ``measure_wedge_sign`` to check other data.

Weights (``wedge_weights``) are in the *reference* frame of a particle with
matrix M (tomo -> reference): w_ref(k) = w_tomo(M.T @ k), returned on the
``np.fft.rfftn`` grid of a cubic box.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CTFParams:
    """Microscope + per-tilt defocus (Warp convention: defocus in um, positive = underfocus)."""
    defocus_um: np.ndarray          # per used tilt, same order as TiltInfo.angles
    voltage_kv: float = 300.0
    cs_mm: float = 2.7
    amplitude: float = 0.07
    phase_shift_rad: float = 0.0


@dataclass
class TiltInfo:
    angles: np.ndarray                 # degrees, used tilts only
    dose: np.ndarray | None = None     # accumulated e-/A^2 per tilt (same order)
    sign: float = 1.0
    source: str = ""
    meta: dict = field(default_factory=dict)
    ctf: CTFParams | None = None

    @property
    def range(self):
        b = self.sign * self.angles
        return float(b.min()), float(b.max())

    # ---------------------------------------------------------------- loaders
    @classmethod
    def from_warp_xml(cls, path, sign: float | None = None) -> "TiltInfo":
        raw = Path(path).read_bytes()
        enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
        root = ET.fromstring(raw.decode(enc))

        def arr(tag, conv=float):
            el = root.find(tag)
            return None if el is None or not (el.text or "").strip() else np.array([conv(v) for v in el.text.split()])

        ang = arr("Angles")
        if ang is None:
            raise ValueError(f"{path}: no <Angles>")
        use = arr("UseTilt", lambda s: s.strip().lower() == "true")
        use = np.ones(len(ang), bool) if use is None else use
        dose = arr("Dose")
        inverted = root.attrib.get("AreAnglesInverted", "False").lower() == "true"
        if sign is None:
            sign = -1.0 if inverted else 1.0
        return cls(ang[use], None if dose is None else dose[use], sign, str(path),
                   {"AreAnglesInverted": inverted, "PlaneNormal": root.attrib.get("PlaneNormal")},
                   _warp_ctf(root, len(ang), use))

    @classmethod
    def from_tlt(cls, path, sign: float = 1.0) -> "TiltInfo":
        """i3/protomo ``TILT ANGLE x`` lines, or IMOD .tlt (one angle per line)."""
        txt = Path(path).read_text()
        hits = re.findall(r"TILT ANGLE\s+(-?[\d.]+)", txt)
        if hits:
            ang = np.array(hits, float)
        else:
            ang = np.array([float(l.split()[0]) for l in txt.splitlines() if l.strip()])
        return cls(ang, None, sign, str(path))

    @classmethod
    def from_range(cls, lo: float, hi: float, step: float = 3.0, sign: float = 1.0) -> "TiltInfo":
        ang = np.arange(lo, hi, step)
        ang = np.append(ang, hi)  # endpoints always included
        return cls(ang, None, sign, f"range {lo}..{hi}/{step}")


def _warp_ctf(root, n_tilts: int, use) -> CTFParams | None:
    """CTF block + per-tilt GridCTF defocus from a Warp tilt-series XML (None if absent)."""
    ctf = root.find("CTF")
    if ctf is None:
        return None
    par = {p.attrib.get("Name"): p.attrib.get("Value") for p in ctf}
    base = float(par.get("Defocus", "nan"))
    grid = root.find("GridCTF")
    dz = np.full(n_tilts, base)
    if grid is not None and len(grid):
        nodes = sorted(grid, key=lambda n: int(n.attrib.get("Z", 0)))
        vals = np.array([float(n.attrib["Value"]) for n in nodes])
        if len(vals) == n_tilts:
            dz = vals
        elif len(vals) == 1:
            dz = np.full(n_tilts, vals[0])
    if not np.isfinite(dz).all():
        return None
    return CTFParams(dz[use], float(par.get("Voltage", 300)), float(par.get("Cs", 2.7)),
                     float(par.get("Amplitude", 0.07)), float(par.get("PhaseShift", 0)) * np.pi)


def electron_wavelength(kv: float) -> float:
    """Relativistic electron wavelength in Angstrom."""
    v = kv * 1e3
    return 12.2643 / np.sqrt(v * (1 + 0.978466e-6 * v))


def particle_defocus(tilt: TiltInfo, pos_rel_A, height_sign: float = 1.0) -> np.ndarray:
    """Defocus (A, underfocus positive) of a particle in every tilt.

    pos_rel_A: particle position (x, y, z) relative to the tomogram centre, in A.
    In tilt t the beam direction (tomogram frame) is n = (-sin b, 0, cos b),
    b = sign * angle; the particle's height along the beam, h = p . n, shifts
    the tilt's defocus by ``height_sign * h``. Which sign is right depends on the
    reconstruction's handedness: check with ``py3sta ctf-check``.
    """
    if tilt.ctf is None:
        raise ValueError(f"{tilt.source}: no CTF information")
    b = np.radians(tilt.sign * tilt.angles)
    px, _, pz = np.asarray(pos_rel_A, float)
    h = -px * np.sin(b) + pz * np.cos(b)
    return tilt.ctf.defocus_um * 1e4 + height_sign * h


def ctf_values(k_inv_A, defocus_A, ctf: CTFParams):
    """Isotropic CTF at spatial frequency k (1/A). Sign convention: CTF(k->0) = -amplitude."""
    lam = electron_wavelength(ctf.voltage_kv)
    k2 = k_inv_A ** 2
    chi = np.pi * lam * defocus_A * k2 - 0.5 * np.pi * ctf.cs_mm * 1e7 * lam ** 3 * k2 ** 2 - ctf.phase_shift_rad
    a = ctf.amplitude
    return -(np.sqrt(1 - a * a) * np.sin(chi) + a * np.cos(chi))


CTF_MODES = ("flip", "wiener", "amplitude")


def ctf_transfer(c, mode: str):
    """Per-voxel transfer for the given correction mode (c = CTF, positive at low frequency)."""
    if mode == "flip":
        return np.where(c < 0, -1.0, 1.0)
    if mode == "wiener":
        return c
    if mode == "amplitude":
        return np.abs(c)
    raise ValueError(f"ctf_mode must be one of {CTF_MODES}, not {mode!r}")


def rfft_grid(box: int):
    """kx, ky, kz in cycles/voxel on the rfftn grid of a (box,box,box) array [z,y,x]."""
    k = np.fft.fftfreq(box).astype(np.float32)
    kx = np.fft.rfftfreq(box).astype(np.float32)
    kz, ky, kxx = np.meshgrid(k, k, kx, indexing="ij")
    return kxx, ky, kz


def critical_exposure(k_inv_A):
    """Grant & Grigorieff (2015) critical exposure, e-/A^2, k in 1/A."""
    k = np.maximum(k_inv_A, 1e-4)
    return 0.245 * k ** -1.665 + 2.81


def wedge_weights(box: int, tilt: TiltInfo, matrix=None, apix: float | None = None,
                  dose_weight: bool = False, soft_deg: float = 2.0, grid=None,
                  defocus_A=None, ctf_mode: str | None = None) -> np.ndarray:
    """Tilt-coverage weights on the rfft grid, in the reference frame of ``matrix``.

    Binary wedge with a ``soft_deg`` cosine edge; with ``dose_weight`` (needs
    tilt.dose and apix) each point gets the exposure filter of its nearest tilt.

    CTF (needs tilt.ctf, apix and the particle's per-tilt ``defocus_A`` from
    ``particle_defocus``): each Fourier voxel gets the CTF of its nearest tilt
    at |k| (3D CTF model of tilt-series reconstructions).
    ctf_mode (see ``CTF_MODES``):
      "flip"      sign(CTF)  - tomogram NOT CTF-corrected (plain WBP, e.g. IMOD/etomo)
      "wiener"    CTF        - tomogram NOT CTF-corrected; also restores amplitudes
      "amplitude" |CTF|      - tomogram already phase-flipped (Warp, novaCTF): amplitude weighting only
    """
    kx, ky, kz = grid if grid is not None else rfft_grid(box)
    if matrix is not None:
        m = np.asarray(matrix, np.float32)
        # k_tomo = M.T @ k_ref  ->  components: sum_j M[j, i] * k_j
        kx, ky, kz = (m[0, i] * kx + m[1, i] * ky + m[2, i] * kz for i in range(3))
    # fold Friedel mates so kx >= 0, then elevation in (-90, 90]
    flip = kx < 0
    kx_f = np.where(flip, -kx, kx)
    kz_f = np.where(flip, -kz, kz)
    beta = np.degrees(np.arctan2(kz_f, kx_f))
    on_axis = (np.abs(kx_f) < 1e-6) & (np.abs(kz_f) < 1e-6)
    lo, hi = tilt.range
    if soft_deg > 0:
        d = np.minimum(beta - lo, hi - beta)  # >0 inside
        w = np.clip(0.5 + d / soft_deg, 0, 1)
        w = 0.5 - 0.5 * np.cos(np.pi * w)
    else:
        w = ((beta >= lo) & (beta <= hi)).astype(np.float32)
    if dose_weight or ctf_mode:
        if apix is None:
            raise ValueError("dose / CTF weighting needs apix")
        order = np.argsort(tilt.sign * tilt.angles)
        b_t = (tilt.sign * tilt.angles)[order]
        j = np.clip(np.searchsorted(b_t, beta), 1, len(b_t) - 1)
        nearest = np.where(np.abs(beta - b_t[j - 1]) < np.abs(beta - b_t[j]), j - 1, j)
        # on the tilt axis (ky only) every tilt samples k: use the 0-tilt
        nearest = np.where(on_axis, np.argmin(np.abs(b_t)), nearest)
        kmag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2) / apix
    if dose_weight:
        if tilt.dose is None:
            raise ValueError("dose weighting needs tilt.dose")
        w = np.where(on_axis, 1.0, w) * np.exp(-tilt.dose[order][nearest] / (2 * critical_exposure(kmag)))
    else:
        w = np.where(on_axis, 1.0, w)
    if ctf_mode:
        if tilt.ctf is None or defocus_A is None:
            raise ValueError("CTF weighting needs tilt.ctf and defocus_A")
        # sign chosen so that low frequencies keep their contrast (c -> +amplitude at k -> 0)
        c = -ctf_values(kmag, np.asarray(defocus_A, float)[order][nearest], tilt.ctf)
        w = w * ctf_transfer(c, ctf_mode)
    return w.astype(np.float32)


def measure_wedge_sign(tomo, tilt: TiltInfo, box: int = 64, n: int = 200, seed: int = 0) -> dict:
    """Compare the tomogram's power spectrum with both wedge signs; returns scores and best sign."""
    from .average import extract

    rng = np.random.default_rng(seed)
    nz, ny, nx = tomo.shape
    m = box // 2 + 8
    win = np.hanning(box).astype(np.float32)
    win3 = win[:, None, None] * win[None, :, None] * win[None, None, :]
    P = np.zeros((box, box, box // 2 + 1))
    for _ in range(n):
        c = [rng.uniform(m, nx - m), rng.uniform(m, ny - m), rng.uniform(m, nz - m)]
        s, _ = extract(tomo, c, np.eye(3), box)
        P += np.abs(np.fft.rfftn((s - s.mean()) * win3)) ** 2
    kx, ky, kz = rfft_grid(box)
    kr = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    shell = (kr > 0.08) & (kr < 0.4)
    logp = np.log(P[shell] + 1e-12)
    scores = {}
    for s in (1.0, -1.0):
        w = wedge_weights(box, TiltInfo(tilt.angles, None, s), soft_deg=0)[shell] > 0.5
        scores[s] = float(logp[w].mean() - logp[~w].mean())  # contrast in/out of predicted coverage
    best = max(scores, key=scores.get)
    return {"scores": scores, "best_sign": best}
