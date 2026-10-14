"""i3 ``.trf`` files.

Columns (whitespace separated, no header, no blank lines allowed by i3):
  1      name
  2-4    x y z        integer voxel position in the i3 tomogram
  5-7    dx dy dz     sub-voxel / refined shift (true centre = x + dx, as in trf2star.py)
  8-16   m11..m33     row-major rotation, tomogram -> reference
  17     score        (i3datalist output only)
  18     class        (i3datalist -cls output only)
  19     extra        (i3datalist output; meaning unknown, preserved)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MAT_COLS = [f"m{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
BASE_COLS = ["name", "x", "y", "z", "dx", "dy", "dz", *MAT_COLS]
OPT_COLS = ["score", "cls", "extra"]


def read_trf(path) -> pd.DataFrame:
    path = Path(path)
    rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    if not rows:
        return pd.DataFrame(columns=BASE_COLS)
    ncol = {len(r) for r in rows}
    if len(ncol) != 1:
        raise ValueError(f"{path}: inconsistent column counts {sorted(ncol)}")
    n = ncol.pop()
    if not 16 <= n <= 19:
        raise ValueError(f"{path}: expected 16-19 columns, found {n}")
    df = pd.DataFrame(rows, columns=(BASE_COLS + OPT_COLS)[:n])
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c == "extra" else df[c].astype(float)
    if "cls" in df:
        df["cls"] = df["cls"].astype(int)
    return df


def matrices(df: pd.DataFrame) -> np.ndarray:
    return df[MAT_COLS].to_numpy(float).reshape(-1, 3, 3)


def write_trf(df: pd.DataFrame, path) -> None:
    """Write a 16-column input trf. Rejects duplicate positions (i3 fails on them)."""
    pos = df[["x", "y", "z"]].round().astype(int)
    dup = pos.duplicated(keep=False)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} particles share identical integer x,y,z; i3 cannot handle duplicates")
    lines = []
    for r, p in zip(df.itertuples(index=False), pos.itertuples(index=False)):
        m = " ".join(f"{getattr(r, c):.10f}" for c in MAT_COLS)
        lines.append(f"{r.name} {p.x} {p.y} {p.z} {r.dx:.4f} {r.dy:.4f} {r.dz:.4f} {m}")
    # no trailing blank line: i3 chokes on empty lines
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))
