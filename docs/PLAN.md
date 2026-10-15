# py3sta — Plan

Goal: i3's supervised "alignment through classification" cycle, GPU-accelerated, scriptable, with
modern classifiers and interactive selection, reading/writing RELION 4 STAR (Warp-compatible).

Status of each item lives in `PROGRESS.md`. Phases are ordered; each ends with an acceptance check on
real data (MIM019) and synthetic tests.

---

## 1. Project and cycle structure

```
project/
  project.yaml               # tomogram source, box, pixel size, CTF mode, masks, device, jobs
  input.star                 # starting particles (rlnMicrographName -> tomogram, rlnWarpXMLfile -> tilts/CTF)
  cycle-000/
    cycle.yaml               # parameters of this cycle (copied from previous cycle by `next`, then edited)
    00_input.star            # = previous cycle's _final.star (cycle 0: input.star)
    01_average/              # global average, half maps, FSC
    02_reference/            # references.mrcs + references.star (1..N references)
    03_stack/                # stack.mrcs (+ optional binned msa_stack.mrcs) + stack.star
    04_align/                # aligned.star (optional MRA)
    05_msa/                  # eigenvolumes, scree, scores.npy (+ score columns in star)
    06_classify/             # class_K004.star ... (one per K and method), class averages, counts, dendrogram
    07_embed/                # UMAP/t-SNE embeddings, user-drawn selections (lasso groups)
    08_select/               # selection.yaml (chosen K / groups, per-class action + target), class-average alignment
    cycle-000_final.star     # becomes cycle-001/00_input.star
```

- Every step reads only its upstream outputs and writes its own folder + `manifest.json`
  (parameters, input hashes, versions, timing). Re-running a step overwrites its folder and marks
  downstream steps stale. `py3sta status` shows done/stale/missing.
- Config: YAML, validated (pydantic or dataclasses + explicit checks), clear error messages, units in Å.

## 2. Steps of one cycle

1. **Average** — existing `average_particles` on `00_input.star` (wedge, dose, CTF per config).
2. **Reference(s)** — mode `global` | `classes` (selected class averages of previous cycle) | `file` (one or
   more MRCs). Masked (sphere/ellipsoid/box/file, soft edge) and band-passed (Å) → reference stack.
   Multi-reference = N > 1.
3. **Stack** — extract every particle once, pre-rotated/shifted by its current pose → memory-mapped
   `stack.mrcs` (float16 default) + `stack.star` (row ↔ particle). All later steps read the stack (as i3),
   so alignment searches small deltas around identity. Wedge/CTF weights are recomputed from geometry.
   Optional smaller/binned MSA stack (i3 `MSAIMGSIZE`).
4. **Align (optional MRA)** — stack vs reference(s); result = delta pose composed onto current pose →
   `aligned.star`. Can be disabled (translation-only or skip, like i3 practice).
5. **MSA / dimensionality reduction**
   - **PCA** (randomized SVD, GPU): masked, band-passed stack; default *wedge-masked differences*
     (particle − wedge-applied average) so missing-wedge factors disappear; plain PCA available for i3 parity.
   - Outputs: eigenvolumes (montage + MRC), scree plot, per-particle scores (`py3staPC1..n` columns + `.npy`).
6. **Classification** — on chosen factors (i3 `CLSFACT`), for a list of K:
   - **k-means** (GPU), **Ward hierarchical** (i3 parity, dendrogram), **GMM** (BIC suggestion), HDBSCAN later.
   - Per K: `class_K###.star`, wedge-weighted class averages, counts, montage, dendrogram.
   - Stability score (bootstrap ARI) per K to flag noise-driven classes.
7. **Embedding + interactive selection (UMAP)**
   - **UMAP** (umap-learn; cuML if available) and t-SNE on the top PCA scores → 2D coordinates per particle.
   - **Interactive explorer** `py3sta explore cycle-000`: scatter plot of UMAP (or any two PCs), coloured by
     k-means class / PC score / CC / tomogram; **draw lasso/polygon regions to define particle groups**;
     groups are saved to `07_embed/groups.json` and become a "classification" like any K
     (`class_lasso.star`), usable in `select`.
   - v1: self-contained HTML (plotly.js lasso, data embedded, no server) → "Export groups" writes JSON.
   - v2: local server (FastAPI/Dash) that computes the class average of a drawn group live (GPU) and can
     push it to ChimeraX (REST `remotecontrol`), plus eigenvolume slider views.
8. **Select + align class averages** — `py3sta select`:
   - choose a classification (a K, a method, or the lasso groups),
   - per class: `align` | `leave` | `exclude`,
   - targets: aligned classes **to each other** (exhaustive search among them, then refine against their
     running average) or **to a reference MRC**; targets can be mixed per class group,
   - class transforms are composed onto every member's pose; `leave` keeps poses;
   - `exclude` sets `py3staExcluded = 1` on the members (see "Excluded particles" below).
9. **Final** — `cycle-NNN_final.star`: new angles/origins (RELION 4), `rlnClassNumber`, `py3staCycle`,
   scores; summary report.
10. **Next** — `py3sta next` creates `cycle-NNN+1/`, copies `cycle.yaml`, links `00_input.star`.

### Excluded particles

- Excluded particles stay in every STAR file with `py3staExcluded = 1` (plus `py3staExcludedCycle`, the
  cycle where it was set) and are carried over to all later cycles.
- Every step (average, reference, stack, align, MSA, classify, explore, select) skips rows with
  `py3staExcluded = 1`; they keep their last pose untouched.
- They stay excluded until the user changes the flag back: `py3sta flag cycle-003 --include <selection>`
  (selection by class of a given cycle, lasso group, particle names, or a STAR file), or by editing the column.
- `py3sta flag` also supports `--exclude` for manual exclusion; `status` reports active / excluded counts.

### CLI sketch

```
py3sta init    project/ --star input.star --box 72
py3sta run     cycle-000 [--from stack] [--to classify]
py3sta explore cycle-000                      # UMAP / PCA scatter, lasso groups
py3sta select  cycle-000 --use kmeans:6 --align 1,2,4 --leave 3 --exclude 5 --target each   # or --target ref.mrc
py3sta select  cycle-000 --use lasso --align all --target each
py3sta flag    cycle-003 --include class:cycle-001:5     # revive previously excluded particles
py3sta next    cycle-000
py3sta status
```

## 3. Core engines

- **Alignment engine** (used by MRA and class-average alignment): wedge-constrained normalized
  cross-correlation (score only over Fourier regions sampled by both), CTF-aware weights, FFT translational
  search with per-axis limits and sub-voxel peak; angular grids: i3 cone/spin (`nut nstep spin sstep`)
  around current pose, uniform global grid, coarse→fine; optional membrane-normal restraint from dipole
  priors. Rotate the reference (shared across particles) → batched GPU. Gold-standard halves kept separate.
- **Pose composition**: delta (R, t) in reference frame → new particle matrix and centre; one function,
  tested with synthetic known transforms (both for MRA and class-average propagation).
- **Masks/filters**: sphere, ellipsoid, box, cylinder, file; soft edges; band-pass in Å.

## 4. Phases and acceptance checks

| Phase | Content | Acceptance |
|---|---|---|
| 0 ✅ | I/O, converters, extraction, wedge/dose/CTF averaging, parallel CPU, torch mirror | see PROGRESS.md |
| 1 | Pipeline skeleton: project/cycle dirs, configs, manifests, status, references (global/file), stack, pose composition, `_final.star`, `next`; dummy classification from existing `rlnClassNumber` | stack from `final.star` == stack from original; synthetic class transforms recovered exactly |
| 2 | Alignment engine (CPU + GPU), MRA step | MIM019 angles perturbed ±15° recovered (median < 5°), FSC not worse; 265 ptcl × 2000 orientations < 1 min on GPU |
| 3 | PCA (wedge-masked differences + plain), k-means, Ward, GMM, multi-K outputs | synthetic 2-state data separated; wedge factors absent |
| 4 | UMAP + `explore` (static HTML lasso → groups.json → class_lasso.star) | groups round-trip into `select`; lasso on synthetic data reproduces true classes |
| 5 | `select`: each/reference targets, leave/exclude (flag carried over), `flag --include/--exclude`, class-average alignment, propagate to members, full cycle | one full cycle on MIM019; next-cycle average ≥ previous (FSC, CC vs template) |
| 6 | Reports (HTML per cycle), ChimeraX push, live explorer server (v2) | — |
| 7 | CTF in alignment/MSA scores, multi-tomogram datasets (height sign), novaCTF comparison | — |

## 5. Open questions (ask the user)

- Stack precision default (float16 vs float32) once real dataset sizes are known.
- Reference for class-average alignment when `each`: best-SNR class vs running average (default: running average).
