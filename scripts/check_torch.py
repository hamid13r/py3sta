"""Check the PyTorch/CUDA backend against the numpy reference and time both.

    python scripts/check_torch.py STAR TOMOGRAM [--tilt XML] [--box 72] [--n 100]

Prints max differences for: extraction, wedge weights, plain average, wedge-weighted average.
"""
import argparse
import time

import numpy as np
import torch

from py3 import average as A
from py3 import backend_torch as T
from py3.io.mrc import read_mrc
from py3.io.particles import ParticleSet
from py3.wedge import TiltInfo, wedge_weights


def rel(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("star")
    ap.add_argument("tomogram")
    ap.add_argument("--tilt")
    ap.add_argument("--box", type=int, default=72)
    ap.add_argument("--n", type=int, default=100, help="particles used for the comparison")
    a = ap.parse_args()

    dev = T.default_device()
    print(f"torch {torch.__version__}, device {dev}"
          + (f" ({torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda})" if dev.type == "cuda" else ""))
    ps = ParticleSet.from_star(a.star)
    ps.df = ps.df.iloc[: a.n].reset_index(drop=True)
    tomos = A.map_tomograms(ps, [a.tomogram])
    vol, apix = read_mrc(a.tomogram)
    c = ps.centers_px[:8] * ps.ts_apix / apix
    M = ps.matrices[:8]

    sub_t, ins_t = T.extract_batch(vol, c, M, a.box, dev)
    sub_t, ins_t = sub_t.cpu().numpy(), ins_t.cpu().numpy()
    err = max(rel(sub_t[i], A.extract(vol, c[i], M[i], a.box)[0]) for i in range(8))
    print(f"extraction            max rel diff {err:.2e}")

    tilts = None
    if a.tilt:
        t = TiltInfo.from_warp_xml(a.tilt) if a.tilt.endswith(".xml") else TiltInfo.from_tlt(a.tilt)
        tilts = {next(iter(tomos)): t}
        for dose in (False, True) if t.dose is not None else (False,):
            wt = T.wedge_weights_batch(a.box, t, M, apix, dose, device=dev).cpu().numpy()
            e = max(rel(wt[i], wedge_weights(a.box, t, M[i], apix, dose)) for i in range(8))
            print(f"wedge weights dose={dose!s:5s} max rel diff {e:.2e}")

    for label, kw in [("plain", {}), ("wedge", {"tilts": tilts})] if tilts else [("plain", {})]:
        t0 = time.time(); rn = A.average_particles(ps, tomos, a.box, **kw); tn = time.time() - t0
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time(); rt = T.average_particles_torch(ps, tomos, a.box, device=dev, **kw)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        tt = time.time() - t0
        print(f"average {label:6s}  max rel diff {rel(rt.average, rn.average):.2e}   "
              f"numpy {tn:.1f}s  torch {tt:.1f}s  ({rn.n_used} particles)")


if __name__ == "__main__":
    main()
