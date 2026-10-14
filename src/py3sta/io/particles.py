"""ParticleSet: the in-memory particle table, stored with RELION 4 semantics.

Internal convention (identical to a RELION 4 tomography particles.star):
  rlnTomoName                    tomogram / tilt-series name (no extension)
  rlnCoordinateX/Y/Z             pixels at the tilt-series pixel size ``ts_apix``,
                                 0 = centre of the first voxel (Warp convention)
  rlnOriginX/Y/ZAngst            shifts in Angstrom; true centre = coord - origin/ts_apix
  rlnAngleRot/Tilt/Psi           ZYZ Euler angles (see geometry.py)

Coordinates from a tomogram at pixel size ``tomo_apix`` are converted with
``coord_ts = coord_tomo * tomo_apix / ts_apix``. This is how Warp/WarpTools
relates tomograms of different binning. Tomograms binned with IMOD ``binvol``
instead have an extra (bin-1)/2 voxel offset; pass ``offset`` if that applies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..geometry import euler_to_matrix, matrix_to_euler
from . import trf as trfio
from .star import read_star, write_star

XYZ = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
ORI = ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
ANG = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
SUBTOMO_ANG = ["rlnTomoSubtomogramRot", "rlnTomoSubtomogramTilt", "rlnTomoSubtomogramPsi"]

DEFAULT_TOMO_REGEX = r"\.mrc.*$"  # 'TS_01.mrc_13.30Apx.mrc' -> 'TS_01'


def tomo_name_from(name: str, strip_regex: str | None = DEFAULT_TOMO_REGEX) -> str:
    name = Path(str(name)).name
    return re.sub(strip_regex, "", name) if strip_regex else name


@dataclass
class ParticleSet:
    df: pd.DataFrame
    ts_apix: float
    voltage: float = 300.0
    cs: float = 2.7
    amplitude_contrast: float = 0.1
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    def __post_init__(self):
        for c in ORI + ANG:
            if c not in self.df:
                self.df[c] = 0.0
        missing = [c for c in ["rlnTomoName", *XYZ] if c not in self.df]
        if missing:
            raise ValueError(f"particle table missing columns {missing}")
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    @property
    def centers_px(self) -> np.ndarray:
        """True particle centres in tilt-series pixels (origins applied)."""
        return self.df[XYZ].to_numpy(float) - self.df[ORI].to_numpy(float) / self.ts_apix

    @property
    def matrices(self) -> np.ndarray:
        return euler_to_matrix(*self.df[ANG].to_numpy(float).T)

    # --------------------------------------------------------------- i3 trf
    @classmethod
    def from_trf(cls, paths, tomo_apix: float, ts_apix: float | None = None,
                 tomo_names: dict | None = None, strip_regex: str | None = DEFAULT_TOMO_REGEX,
                 offset: float = 0.0, **kw) -> "ParticleSet":
        """Read one or many trf / i3datalist files.

        tomo_names: optional {trf-file-stem or particle name -> tomogram name}.
        Otherwise the tomogram name is the particle name with ``_<7 digits>``
        (py3sta particle index) and ``strip_regex`` removed.
        """
        ts_apix = ts_apix or tomo_apix
        s = tomo_apix / ts_apix
        frames = []
        for p in ([paths] if isinstance(paths, (str, Path)) else paths):
            t = trfio.read_trf(p)
            if t.empty:
                continue
            stem = Path(p).stem
            if tomo_names and stem in tomo_names:
                tomo = pd.Series(tomo_names[stem], index=t.index)
            else:
                base = t["name"].str.replace(r"_\d{7}$", "", regex=True)
                tomo = base.map(lambda n: tomo_names.get(n, tomo_name_from(n, strip_regex))
                                if tomo_names else tomo_name_from(n, strip_regex))
            eul = matrix_to_euler(trfio.matrices(t))
            out = pd.DataFrame({
                "rlnTomoName": tomo,
                "rlnCoordinateX": (t["x"] + offset) * s,
                "rlnCoordinateY": (t["y"] + offset) * s,
                "rlnCoordinateZ": (t["z"] + offset) * s,
                # true = x + dx  (trf2star.py)  <=>  RELION origin = -dx (in Angstrom)
                "rlnOriginXAngst": -t["dx"] * tomo_apix,
                "rlnOriginYAngst": -t["dy"] * tomo_apix,
                "rlnOriginZAngst": -t["dz"] * tomo_apix,
                "rlnAngleRot": eul[:, 0], "rlnAngleTilt": eul[:, 1], "rlnAnglePsi": eul[:, 2],
                "i3Name": t["name"],
            })
            if "score" in t:
                out["i3Score"] = t["score"]
            if "cls" in t:
                out["rlnClassNumber"] = t["cls"]
            frames.append(out)
        if not frames:
            raise ValueError("no particles found in the given trf files")
        return cls(pd.concat(frames, ignore_index=True), ts_apix, **kw)

    def to_trf(self, out_dir, tomo_apix: float, offset: float = 0.0,
               name_fmt: str = "{tomo}_{idx:07d}") -> dict[str, Path]:
        """Write one trf per tomogram into ``out_dir``; returns {tomo: path}.

        Integer part goes into x,y,z and the fractional remainder into dx,dy,dz,
        so no sub-voxel precision is lost (pos2i3.py truncated with int()).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        true = self.centers_px * self.ts_apix / tomo_apix - offset
        ip = np.round(true)
        frac = true - ip
        mats = self.matrices.reshape(-1, 9)
        written = {}
        for tomo, idx in self.df.groupby("rlnTomoName", sort=False).groups.items():
            idx = np.asarray(idx)
            t = pd.DataFrame({
                "name": [name_fmt.format(tomo=tomo, idx=i) for i in range(len(idx))],
                "x": ip[idx, 0], "y": ip[idx, 1], "z": ip[idx, 2],
                "dx": frac[idx, 0], "dy": frac[idx, 1], "dz": frac[idx, 2],
            })
            for k, c in enumerate(trfio.MAT_COLS):
                t[c] = mats[idx, k]
            path = out_dir / f"{tomo}.trf"
            trfio.write_trf(t, path)
            written[tomo] = path
        return written

    # ------------------------------------------------------------ RELION 4
    def optics(self) -> pd.DataFrame:
        return pd.DataFrame({
            "rlnOpticsGroup": [1],
            "rlnOpticsGroupName": ["opticsGroup1"],
            "rlnSphericalAberration": [self.cs],
            "rlnVoltage": [self.voltage],
            "rlnAmplitudeContrast": [self.amplitude_contrast],
            "rlnTomoTiltSeriesPixelSize": [self.ts_apix],
        })

    def to_relion4(self, path, micrograph_suffix: str | None = ".tomostar") -> None:
        """RELION 4 tomo particles.star (data_optics + data_particles).

        Also writes rlnMicrographName = <tomo><suffix> so WarpTools can match
        tilt series; set ``micrograph_suffix=None`` to omit it.
        """
        df = self.df.copy()
        df["rlnOpticsGroup"] = 1
        if "rlnTomoParticleName" not in df:
            df["rlnTomoParticleName"] = df["rlnTomoName"] + "/" + (df.groupby("rlnTomoName").cumcount() + 1).astype(str)
        if micrograph_suffix is not None:
            df["rlnMicrographName"] = df["rlnTomoName"] + micrograph_suffix
        lead = ["rlnTomoName", "rlnTomoParticleName", *XYZ, *ORI, *ANG]
        cols = lead + [c for c in df.columns if c not in lead]
        # STAR labels must be rln*; keep py3sta bookkeeping columns too (RELION ignores unknown labels)
        write_star({"optics": self.optics(), "particles": df[cols]}, path)

    @classmethod
    def from_star(cls, path, ts_apix: float | None = None, **kw) -> "ParticleSet":
        """Read RELION 4 tomo particles, or a RELION 3/Warp single-block file."""
        blocks = read_star(path)
        if "particles" in blocks:
            df = blocks["particles"].copy()
            optics = blocks.get("optics")
            # pixel size of rlnCoordinate*: RELION 4 tomo label first, then RELION 3.1 / Warp labels
            for lab in ("rlnTomoTiltSeriesPixelSize", "rlnMicrographPixelSize", "rlnImagePixelSize"):
                if ts_apix is None and optics is not None and lab in optics:
                    ts_apix = float(optics[lab].iloc[0])
        else:
            df = next(iter(blocks.values())).copy()
        if ts_apix is None:
            raise ValueError("pixel size unknown: pass ts_apix (no rlnTomoTiltSeriesPixelSize in file)")
        if "rlnTomoName" not in df:
            if "rlnMicrographName" not in df:
                raise ValueError("need rlnTomoName or rlnMicrographName")
            df["rlnTomoName"] = df["rlnMicrographName"].map(lambda n: re.sub(r"\.(tomostar|mrc.*)$", "", Path(str(n)).name))
        if any(c in df for c in SUBTOMO_ANG):
            sub = df.reindex(columns=SUBTOMO_ANG).fillna(0).to_numpy(float)
            if np.abs(sub).max() > 1e-6:
                raise NotImplementedError(
                    "rlnTomoSubtomogram* angles are non-zero; composing them with rlnAngle* is not implemented yet")
        return cls(df, ts_apix, **kw)

    # ----------------------------------------------------------------- Warp
    def to_warp(self, path, suffix: str = ".tomostar") -> float:
        """Minimal single-block STAR for ``WarpTools ts_export_particles``.

        Origins are folded into coordinates. Returns the value to pass as
        ``--coords_angpix``.
        """
        c = self.centers_px
        df = pd.DataFrame({
            "rlnCoordinateX": c[:, 0], "rlnCoordinateY": c[:, 1], "rlnCoordinateZ": c[:, 2],
            "rlnAngleRot": self.df["rlnAngleRot"], "rlnAngleTilt": self.df["rlnAngleTilt"],
            "rlnAnglePsi": self.df["rlnAnglePsi"],
            "rlnMicrographName": self.df["rlnTomoName"] + suffix,
        })
        if "rlnClassNumber" in self.df:
            df["rlnClassNumber"] = self.df["rlnClassNumber"]
        write_star({"": df}, path)
        return self.ts_apix
