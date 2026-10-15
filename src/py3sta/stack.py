"""Particle stack: raw, unrotated crops around every particle (PLAN.md §1 "Particle stack").

    stack/stack.mrcs   float16 cubes of size box + 2*margin, one per row (MRC of shape (n*size, size, size))
    stack/stack.star   data_stack (box, margin, size, apix, dtype) + data_rows (one row per crop)

Crops are cut on whole voxels (no interpolation); the pose is applied when a crop is used
(average.extract_crop / backend_torch.sample_cubes), so the pipeline interpolates once.
A crop stays valid while the particle centre is within ``max_drift(margin)`` voxels (per axis) of the
crop centre: then the extraction sphere (radius box//2 - 1) plus one voxel for trilinear
interpolation lies inside the crop for any rotation. Pose changes never invalidate a crop; a crop is
re-made when its tomogram changed (path, size or mtime) or the particle drifted too far.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .io.mrc import open_volume_stack, read_header, read_mrc, resize_volume_stack
from .io.particles import ParticleSet
from .io.star import read_star, write_star

DATA_FILE = "stack.mrcs"
TABLE_FILE = "stack.star"
NAME = "rlnTomoParticleName"
LO = ["py3staCropX", "py3staCropY", "py3staCropZ"]
TSHAPE = ["py3staTomoNX", "py3staTomoNY", "py3staTomoNZ"]
FLOAT16_MAX = 65000.0


class StackError(ValueError):
    pass


def crop_size(box: int, margin: int) -> int:
    return box + 2 * margin


def max_drift(margin: int) -> int:
    """Largest per-axis distance (voxels) between particle centre and crop centre that keeps the
    extraction sphere inside the crop (checked by tests for even and odd boxes)."""
    return margin - 1


def file_id(path) -> tuple[str, int, int]:
    """(absolute path, size in bytes, mtime in ns): identifies the tomogram a crop was cut from."""
    st = os.stat(path)
    return os.path.abspath(path), st.st_size, st.st_mtime_ns


def particle_centres(ps: ParticleSet, apix: float) -> np.ndarray:
    """True particle centres in tomogram voxels (x, y, z)."""
    return ps.centers_px * (ps.ts_apix / apix)


def particle_names(ps: ParticleSet) -> np.ndarray:
    if NAME not in ps.df:
        raise StackError(f"particles need {NAME} to be matched with stack rows ('py3sta init' adds it)")
    names = ps.df[NAME].astype(str).to_numpy()
    dup = pd.Series(names)[pd.Series(names).duplicated()]
    if len(dup):
        raise StackError(f"{NAME} is not unique: {', '.join(dup.unique()[:5])}")
    return names


@dataclass
class StackLocation:
    """Where each particle of a ParticleSet lives in the stack (arrays aligned with ps rows)."""
    rows: np.ndarray                  # (n,) stack row
    lo: np.ndarray                    # (n, 3) crop origin, tomogram voxels xyz
    tomo_shape: dict                  # {rlnTomoName: (nz, ny, nx)}


@dataclass
class ParticleStack:
    root: Path
    table: pd.DataFrame
    box: int
    margin: int
    apix: float
    dtype: str = "float16"

    @property
    def size(self) -> int:
        return crop_size(self.box, self.margin)

    @property
    def data_path(self) -> Path:
        return self.root / DATA_FILE

    @property
    def nbytes(self) -> int:
        return len(self.table) * self.size ** 3 * np.dtype(self.dtype).itemsize

    @classmethod
    def open(cls, root) -> "ParticleStack":
        root = Path(root)
        if not (root / TABLE_FILE).is_file():
            raise StackError(f"{root}: no stack (run 'py3sta stack' or 'py3sta run')")
        b = read_star(root / TABLE_FILE)
        g = b["stack"].iloc[0]
        table = b.get("rows", pd.DataFrame())
        return cls(root, table, int(g["box"]), int(g["margin"]), float(g["apix"]), str(g["dtype"]))

    def data(self, mode: str = "r"):
        return open_volume_stack(self.data_path, len(self.table), self.size, mode=mode)

    def write_table(self) -> None:
        head = pd.DataFrame([{"box": self.box, "margin": self.margin, "size": self.size,
                              "apix": self.apix, "dtype": self.dtype, "rows": len(self.table)}])
        tmp = self.root / (TABLE_FILE + ".tmp")
        write_star({"stack": head, "rows": self.table}, tmp)
        os.replace(tmp, self.root / TABLE_FILE)

    def locate(self, ps: ParticleSet, shift_limit: int = 0) -> StackLocation:
        """Stack rows for ``ps``; raises if a particle is missing or drifted beyond the crop margin."""
        names = particle_names(ps)
        index = pd.Series(np.arange(len(self.table)), index=self.table[NAME].astype(str))
        missing = [n for n in names if n not in index.index]
        if missing:
            raise StackError(f"{len(missing)} particle(s) not in the stack (refresh it): {', '.join(missing[:5])}")
        rows = index.loc[names].to_numpy()
        lo = self.table[LO].to_numpy(np.int64)[rows]
        drift = np.abs(particle_centres(ps, self.apix) - (lo + self.size // 2)).max(1)
        bad = drift > max_drift(self.margin) - shift_limit
        if bad.any():
            raise StackError(f"{int(bad.sum())} particle(s) drifted beyond the crop margin (refresh the stack): "
                             + ", ".join(names[bad][:5]))
        shapes = {}
        for tomo, r in zip(ps.df["rlnTomoName"], rows):
            if tomo not in shapes:
                shapes[tomo] = tuple(int(v) for v in self.table.loc[r, TSHAPE].to_numpy()[::-1])
        return StackLocation(rows, lo, shapes)


@dataclass
class RefreshReport:
    n_rows: int = 0
    n_new: int = 0
    n_tomogram: int = 0               # re-cropped because the tomogram path / file changed
    n_drift: int = 0                  # re-cropped because the particle drifted
    rebuilt: bool = False             # whole stack re-made (box / margin / apix / dtype changed or new)
    seconds: float = 0.0
    changed_tomograms: list = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.rebuilt or bool(self.n_new or self.n_tomogram or self.n_drift)


def refresh_stack(root, ps: ParticleSet, tomograms: dict, box: int, margin: int, apix: float,
                  dtype: str = "float16", jobs: int = 1, shift_limit: int = 0, progress=None) -> RefreshReport:
    """Create the stack, or bring it up to date with ``ps``: add new particles, re-crop particles whose
    tomogram changed or that drifted more than ``max_drift(margin) - shift_limit``. Crops are cut around
    the current (rounded) centre. Rows of particles no longer in ``ps`` are kept (harmless).

    tomograms: {rlnTomoName: path} (from rlnMicrographName). Writes only what changed.
    """
    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()
    if margin < 1:
        raise StackError(f"stack margin must be >= 1 voxel (got {margin})")
    if max_drift(margin) - shift_limit < 1:            # a fresh crop may be 0.5 voxel off-centre
        raise StackError(f"shift limit {shift_limit} voxel(s) needs a stack margin >= {shift_limit + 2} (is {margin})")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rep = RefreshReport()
    names = particle_names(ps)
    centres = particle_centres(ps, apix)
    tomo_of = ps.df["rlnTomoName"].astype(str).to_numpy()
    ids = {t: file_id(p) for t, p in tomograms.items() if t in set(tomo_of)}
    missing = sorted(set(tomo_of) - set(ids))
    if missing:
        raise StackError(f"no tomogram for {', '.join(missing[:5])}")
    shapes = {t: read_header(ids[t][0])["shape"] for t in ids}
    for t, p in ids.items():
        a = read_header(p[0])["apix"]
        if abs(a - apix) > 1e-3:
            raise StackError(f"{p[0]}: voxel size {a:g} Å differs from the project's {apix:g} Å")

    try:
        st = ParticleStack.open(root)
        if (st.box, st.margin, st.dtype) != (box, margin, dtype) or abs(st.apix - apix) > 1e-4:
            st = None
    except StackError:
        st = None
    size = crop_size(box, margin)
    if st is None:
        rep.rebuilt = True
        st = ParticleStack(root, pd.DataFrame(columns=[NAME, "rlnTomoName", *LO, "py3staTomogram",
                                                       "py3staTomoBytes", "py3staTomoMtimeNs", *TSHAPE]),
                           box, margin, apix, dtype)
        open_volume_stack(st.data_path, 0, size, dtype, apix, mode="w+")

    table = st.table.reset_index(drop=True)
    index = pd.Series(np.arange(len(table)), index=table[NAME].astype(str)) if len(table) else pd.Series(dtype=int)
    lo_new = np.round(centres).astype(np.int64) - size // 2
    todo_rows, todo_idx = [], []                      # stack row, ps index
    new_idx = [i for i, n in enumerate(names) if n not in index.index]
    old_idx = np.array([i for i, n in enumerate(names) if n in index.index], int)
    if len(old_idx):
        rows = index.loc[names[old_idx]].to_numpy()
        cur = table.loc[rows]
        tid = np.array([ids[t] for t in tomo_of[old_idx]], dtype=object)
        changed_tomo = ((cur["py3staTomogram"].astype(str).to_numpy() != np.array([t[0] for t in tid]))
                        | (cur["py3staTomoBytes"].to_numpy(np.int64) != np.array([t[1] for t in tid], np.int64))
                        | (cur["py3staTomoMtimeNs"].to_numpy(np.int64) != np.array([t[2] for t in tid], np.int64)))
        drift = np.abs(centres[old_idx] - (cur[LO].to_numpy(np.int64) + size // 2)).max(1)
        drifted = ~changed_tomo & (drift > max_drift(margin) - shift_limit)
        rep.n_tomogram, rep.n_drift = int(changed_tomo.sum()), int(drifted.sum())
        rep.changed_tomograms = sorted(set(tomo_of[old_idx][changed_tomo]))
        sel = changed_tomo | drifted
        todo_rows += list(rows[sel])
        todo_idx += list(old_idx[sel])
    if new_idx:
        start = len(table)
        add = pd.DataFrame({NAME: names[new_idx], "rlnTomoName": tomo_of[new_idx]})
        table = pd.concat([table, add], ignore_index=True)
        todo_rows += list(range(start, start + len(new_idx)))
        todo_idx += new_idx
        rep.n_new = len(new_idx)
        resize_volume_stack(st.data_path, len(table), size)
    rep.n_rows = len(table)
    if not todo_rows:
        rep.seconds = time.time() - t0
        return rep

    todo_rows, todo_idx = np.asarray(todo_rows, int), np.asarray(todo_idx, int)
    # mark rows invalid first: an interrupted refresh re-crops them next time
    table.loc[todo_rows, "py3staTomoBytes"] = -1
    table.loc[todo_rows, "py3staTomoMtimeNs"] = -1
    for t in set(tomo_of[todo_idx]):
        m = todo_rows[tomo_of[todo_idx] == t]
        table.loc[m, "rlnTomoName"] = t
        table.loc[m, "py3staTomogram"] = ids[t][0]
        table.loc[m, TSHAPE] = list(shapes[t][::-1])
    table.loc[todo_rows, LO] = lo_new[todo_idx]
    st.table = table
    st.write_table()

    tasks = []
    for t in dict.fromkeys(tomo_of[todo_idx]):
        m = tomo_of[todo_idx] == t
        tasks.append((ids[t][0], str(st.data_path), size, todo_rows[m], lo_new[todo_idx[m]], len(table)))
    done = 0
    workers = max(1, min(int(jobs), 8, len(tasks)))   # each worker holds one tomogram slab in RAM
    if workers == 1:
        results = map(_crop_task, *zip(*tasks))
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        results = ex.map(_crop_task, *zip(*tasks))
    try:
        for (path, rows_done), task in zip(results, tasks):
            t = next(k for k, v in ids.items() if v[0] == path)
            table.loc[rows_done, "py3staTomoBytes"] = ids[t][1]
            table.loc[rows_done, "py3staTomoMtimeNs"] = ids[t][2]
            done += len(rows_done)
            if progress:
                progress(done, len(todo_rows))
    finally:
        if workers > 1:
            ex.shutdown()
    for t, (p, nbytes, mtime) in ids.items():                 # tomogram replaced while cropping?
        if t in set(tomo_of[todo_idx]) and file_id(p) != (p, nbytes, mtime):
            raise StackError(f"tomogram {t} changed while the stack was being made; run again")
    st.table = table
    st.write_table()
    rep.seconds = time.time() - t0
    return rep


def _crop_task(tomo_path, data_path, size, rows, lo, n_rows):
    """Worker: cut crops (zero padded at tomogram edges) from one tomogram into stack rows."""
    vol, _ = read_mrc(tomo_path)
    shape = np.array(vol.shape[::-1])
    data = open_volume_stack(data_path, n_rows, size, mode="r+")
    order = np.argsort(lo[:, 2])
    # many particles: read the z-range they need as one sequential slab; few: read through the memmap
    if len(rows) >= 8:
        z0 = int(np.clip(lo[:, 2].min(), 0, shape[2]))
        z1 = int(np.clip(lo[:, 2].max() + size, 0, shape[2]))
        src, zoff = np.asarray(vol[z0:z1]), z0
    else:
        src, zoff = vol, 0
    for k in order:
        l = lo[k]
        out = np.zeros((size,) * 3, np.float32)
        a, e = np.clip(l, 0, shape), np.clip(l + size, 0, shape)
        if np.all(e > a):
            d0 = a - l
            d1 = d0 + (e - a)
            out[d0[2]:d1[2], d0[1]:d1[1], d0[0]:d1[0]] = src[a[2] - zoff:e[2] - zoff, a[1]:e[1], a[0]:e[0]]
        if data.dtype == np.float16 and np.abs(out).max() > FLOAT16_MAX:
            raise StackError(f"{tomo_path}: values exceed the float16 range; use stack dtype float32")
        data[rows[k]] = out
    data.flush()
    return tomo_path, rows
