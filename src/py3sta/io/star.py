"""Minimal, dependency-free STAR reader/writer (loop and key-value blocks)."""
from __future__ import annotations

import shlex
from pathlib import Path

import numpy as np
import pandas as pd


_NAN = {"nan": np.nan, "NaN": np.nan, "-nan": np.nan, "NAN": np.nan}


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except (ValueError, TypeError):
            try:                                       # pandas does not parse "nan" (written for missing values)
                df[c] = pd.to_numeric(df[c].replace(_NAN))
            except (ValueError, TypeError):
                pass
    return df


def read_star(path) -> dict[str, pd.DataFrame]:
    """Return {block_name: DataFrame}. Key-value blocks become one-row frames."""
    blocks: dict[str, pd.DataFrame] = {}
    name, labels, rows, kv, in_loop = None, [], [], {}, False

    def flush():
        if name is None:
            return
        if labels:
            blocks[name] = _to_numeric(pd.DataFrame(rows, columns=labels))
        elif kv:
            blocks[name] = _to_numeric(pd.DataFrame([kv]))
        else:
            blocks[name] = pd.DataFrame()

    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("data_"):
            flush()
            name, labels, rows, kv, in_loop = line[5:], [], [], {}, False
        elif line.startswith("loop_"):
            in_loop = True
        elif line.startswith("_"):
            parts = line.split()
            if in_loop and not rows:
                labels.append(parts[0][1:])
            else:
                kv[parts[0][1:]] = parts[1] if len(parts) > 1 else ""
        elif in_loop:
            vals = shlex.split(line) if ('"' in line or "'" in line) else line.split()
            if len(vals) != len(labels):
                raise ValueError(f"{path}: row has {len(vals)} values, expected {len(labels)}: {raw!r}")
            rows.append(vals)
    flush()
    return blocks


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.6f}"
    s = str(v)
    return f'"{s}"' if (" " in s or s == "") else s


def write_star(blocks: dict[str, pd.DataFrame], path) -> None:
    out = ["", "# written by py3sta", ""]
    for name, df in blocks.items():
        out += [f"data_{name}", "", "loop_"]
        out += [f"_{c} #{i + 1}" for i, c in enumerate(df.columns)]
        for row in df.itertuples(index=False):
            out.append("\t".join(_fmt(v) for v in row))
        out.append("")
    Path(path).write_text("\n".join(out) + "\n")
