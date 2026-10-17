# py3sta — Plan

Goal: i3's supervised "alignment through classification" cycle, GPU-accelerated, scriptable, with
modern classifiers and interactive selection, reading/writing RELION 4 STAR (Warp-compatible).

Status of each item lives in `PROGRESS.md`. Phases are ordered; each ends with an acceptance check on
real data (MIM019) and synthetic tests.

---

## 1. Project and cycle structure

```
project/
  project.yaml               # box, pixel sizes, wedge, stack margin, device, jobs
  input.star                 # starting particles (rlnMicrographName -> tomogram, rlnWarpXMLfile -> tilts/CTF)
  stack/                     # raw unrotated crops, shared by all cycles (see "Particle stack")
    stack.mrcs               # float16, one row per particle, crop = box + 2*margin
    stack.star               # row <-> particle, crop centre (integer voxels), tomogram path + size/mtime
  cycle-000/
    cycle.yaml               # parameters of this cycle incl. CTF mode (copied by `next`, then edited)
    00_input.star            # COPY of the previous cycle's _final.star (cycle 0: input.star); editable
    01_average/              # global average from the stack (pose applied: one interpolation), half maps, FSC, per-tomogram CC
    02_reference/            # references.mrcs + references.star (1..N references)
    03_align/                # aligned.star (optional MRA; reference rotated, particles in tomogram frame)
    04_msa/                  # rotated/masked/binned MSA stack, eigenvolumes, scree, scores
    05_classify/             # class_K004.star ... (one per K and method; class_column.star), classes.mrcs, counts, dendrogram
    06_embed/                # UMAP/t-SNE embeddings, user-drawn selections (lasso groups)
    07_select/               # selected.star, transforms.star, classes_aligned.mrcs (settings: cycle.yaml select:)
    08_final/cycle-000_final.star   # copied to cycle-001/00_input.star by `next`
```

- Every step reads only its upstream outputs and writes its own folder + `manifest.json`
  (parameters, inputs, outputs, run id, versions + git commit, device, timing, status
  running/done/failed). Re-running a step overwrites its folder and marks downstream steps stale.
  `py3sta status` shows done / stale / missing / running / failed / edited.
- Config: YAML, validated (dataclasses + explicit checks), clear error messages, units in Å.
- Running a step whose upstream is stale / missing / failed is refused with a hint
  (`run --from <upstream>`), unless `--force` (decided 2026-10-15).
- Tomogram paths live only in the STAR file (`rlnMicrographName`); to switch reconstructions (raw WBP →
  novaCTF) edit the cycle's `00_input.star` (no helper command, decided 2026-10-15). The CTF mode belongs to
  the reconstruction and is set in `cycle.yaml`.

### Staleness and change tracking (decided 2026-10-15)

- **Parameters:** each manifest stores its slice of `cycle.yaml` / `project.yaml`; any difference → stale.
- **Binary outputs between steps** (MRCs, MSA stacks): run ids. A step records the run id of each
  upstream step it used; upstream re-run → new id → downstream stale. No hashing of large files.
- **STAR files are user-editable.** Writer and readers record a fingerprint (size, mtime, sha1; the hash
  is computed only when size/mtime changed, so `touch`/copy is not an edit).
  - Edited output → producer shows `done (edited: aligned.star)`; every reader of the old version is stale.
  - An edited STAR is re-validated (parses, required columns) before the next step uses it.
  - Re-running a step whose STAR output was edited needs `--force`; the edited version is first saved as
    `<name>.edited-<date>.star`. Edits are never lost silently.
  - `00_input.star` is a copy (not a link), so editing it never changes an upstream cycle.
  - `py3sta flag` is just a recorded STAR edit.
- **External files** (tomograms, Warp XML, reference/mask MRCs, `input.star`): size + mtime. A
  different file at the same tomogram path → those crops are re-made and the cycle's steps are stale.
- **Compute settings** (backend, GPU, jobs) are not parameters: numpy and torch agree to ~1e-6.
- **Contrast:** `invert_contrast` (project.yaml) multiplies densities by −1 when crops are read, so every
  step sees protein white on black (RELION / Warp / ChimeraX convention); the stack keeps raw values.
  `01_average` reports the contrast it sees and warns if protein is dark.
- **Files changed during a run:** a step re-checks the size/mtime of its tomograms at the end; a change →
  `failed: tomogram X changed during the run` (memmapped reads may have mixed old and new data).

### Particle stack (decided 2026-10-15)

Raw, **unrotated** crops around each particle; the pose is applied by each consumer, so there is exactly
one interpolation in the whole pipeline (i3 pre-rotates and interpolates twice after alignment; trilinear
keeps ~80 % amplitude per pass at half Nyquist, ~27 Å on MIM019).

- Integer-voxel crops (plain slicing, no interpolation); the sub-voxel remainder of the centre is applied
  together with the rotation. float16 (decided; check vs float32 on MIM019).
- Crop = box + 2·margin (default margin 4 voxels; MIM019: 80³ → 6.2 GB for 6081 particles vs 4.5 GB at 72³).
  Rotating a box-sized region fills only the inscribed sphere: corners are treated as missing (masks are
  inside the box anyway), like particles at the tomogram edge.
- Project-level and shared by all cycles. Pose changes never make it stale. A crop is re-made only when
  (a) its tomogram path or file changed (STAR edit / new reconstruction), or (b) the particle drifted:
  before a cycle runs, every particle whose |centre − crop centre| + next shift limit > margin is
  re-cropped around its current centre (rows rewritten in place; ~1 ms each). `status` reports counts;
  large drift is also listed as a quality signal (particles sliding onto neighbours / membrane).
- Alignment's translation limit must be ≤ margin (config validation), so no particle leaves its crop
  during a step.
- Alignment compares the reference rotated by each pose with the unrotated crop (wedge fixed per
  tomogram in this frame). MSA builds its own rotated, masked, binned stack from the crops per cycle
  (one interpolation; i3 `MSAIMGSIZE`).

## 2. Steps of one cycle

0. **Stack refresh** (project level, before each cycle) — create `stack/` or re-crop changed/drifted particles.
1. **Average** — poses of `00_input.star` applied to the crops (wedge, dose, CTF per `cycle.yaml`);
   half maps, FSC, per-tomogram CC (each tomogram's average vs the others', flags outliers).
2. **Reference(s)** — mode `global` | `classes` (selected class averages of previous cycle) | `file` (one or
   more MRCs or MRC stacks, fitted to the box, optional `invert`). Masked (sphere/ellipsoid/box/file, soft
   edge) and band-passed (Å) → `references.mrcs` + `references.star` (CC with the current average; negative
   CC warns about inverted contrast). Multi-reference = N > 1; full maps (no half-set references, decided).
3. **Align (optional MRA)** — crops vs reference(s) rotated by each pose, searching small deltas around the
   current pose; result = delta pose composed onto current pose → `aligned.star`. With N references every
   particle is aligned to all of them and keeps the best CC (reference index → `rlnBestReference`, score →
   `py3staRefCC`; `rlnClassNumber` stays for classification). Can be disabled (translation-only or skip, like i3 practice).
4. **MSA / dimensionality reduction** — on a per-cycle rotated, masked, binned MSA stack made from the crops.
   - **PCA** (randomized SVD, GPU): masked, band-passed stack; default *wedge-masked differences*
     (particle − wedge-applied average) so missing-wedge factors disappear; plain PCA available for i3 parity.
   - Outputs: eigenvolumes (montage + MRC), scree plot, per-particle scores (`py3staPC1..n` columns + `.npy`).
   - (done) `msa:` mode `wmd` | `plain` (+ `wedge`: each particle filtered with its own wedge); factor
     diagnostics (adjusted correlation with wedge orientation / tomogram / height / CC) + PNGs.
5. **Classification** — on chosen factors (i3 `CLSFACT`), for a list of K:
   - **k-means** (GPU), **Ward hierarchical** (i3 parity, dendrogram), **GMM** (BIC suggestion), HDBSCAN later.
   - Per K: `class_K###.star`, wedge-weighted class averages, counts, montage, dendrogram.
   - Stability score (bootstrap ARI) per K to flag noise-driven classes.
   - (done) `classify:` method, factors, k, `use` (the K passed to select), bootstrap; per-class Jaccard;
     PNGs (scatter, class montages, stability/BIC, dendrogram). HDBSCAN not yet.
6. **Embedding + interactive selection (UMAP)**
   - **UMAP** (umap-learn; cuML if available) and t-SNE on the top PCA scores → 2D coordinates per particle.
   - **Interactive explorer** `py3sta explore cycle-000`: scatter plot of UMAP (or any two PCs), coloured by
     k-means class / PC score / CC / tomogram; **draw lasso/polygon regions to define particle groups**;
     groups are saved to `06_embed/groups.json` and become a "classification" like any K
     (`class_lasso.star`), usable in `select`.
   - v1: self-contained HTML (plotly.js lasso, data embedded, no server) → "Export groups" writes JSON.
   - (done) `06_embed` (umap | tsne | pca of `embed.factors`, upstream msa only); `py3sta explore` inlines
     plotly.js (offline page); exported groups → `cycle-NNN/groups/NAME.json` → classify run
     `{method: groups, file: groups/NAME.json}` (class 0 = in no group) → compare / `use` like any run.
   - v2: local server (FastAPI/Dash) that computes the class average of a drawn group live (GPU) and can
     push it to ChimeraX (REST `remotecontrol`), plus eigenvolume slider views.
   - Placeholder (done): `classify.method: column` — classes from an integer STAR column, class averages.
7. **Select + align class averages** — `select:` in `cycle.yaml` (done: align/leave/exclude, target reference
   or MRC; `py3sta select` will write that section):
   - choose a classification (a K, a method, or the lasso groups),
   - per class: `align` | `leave` | `exclude`,
   - targets: aligned classes **to each other** (exhaustive search among them, then refine against their
     running average) or **to a reference MRC**; targets can be mixed per class group,
   - class transforms are composed onto every member's pose; `leave` keeps poses;
   - `exclude` sets `py3staExcluded = 1` on the members (see "Excluded particles" below).
8. **Final** — `08_final/cycle-NNN_final.star`: poses from the last pose-writing step (select, else align,
   else the input), every particle kept, `rlnClassNumber`, `py3staCycle`, scores; summary report. (done, no
   class numbers / report yet)
9. **Next** — `py3sta next` creates `cycle-NNN+1/`, copies `cycle.yaml` (only `cycle:` changed) and the final
   STAR as `00_input.star` (a copy). (done)

### Excluded particles

- Excluded particles stay in every STAR file with `py3staExcluded = 1` (plus `py3staExcludedCycle`, the
  cycle where it was set) and are carried over to all later cycles.
- Every step (average, reference, stack, align, MSA, classify, explore, select) skips rows with
  `py3staExcluded = 1`; they keep their last pose untouched.
- They stay excluded until the user changes the flag back: `py3sta flag cycle-003 --include <selection>`
  (selection by class of a given cycle, lasso group, particle names, or a STAR file), or by editing the column.
- `py3sta flag` also supports `--exclude` for manual exclusion; `status` reports active / excluded counts.
  (done: edits the cycle's `00_input.star`; selections `class:CYCLE:K`, `tomo:NAME`, `excluded:CYCLE`,
  `file:PATH`, `all`; `--dry-run`; log in `cycle-NNN/flag.log`. Lasso groups come with the explorer.)

### CLI sketch

```
py3sta init    project/ --star input.star --box 72
py3sta run     cycle-000 [--from average] [--to classify]
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
- **Map-to-map alignment** (done, `align.py`, `py3sta align-maps`): global SO(3) grid + FFT shift search on binned
  maps, local grids, 6-D polish (rotation + sub-voxel shift on the exact-pose CC), hand check, contrast check.
  Particle MRA reuses the masked-CC-over-shifts core with the reference rotated per pose.
- **Pose composition** (done, `poses.py`): delta (R, t) in reference frame → M' = R·M, c' = c + M'ᵀt; one
  function for MRA and class-average propagation; `align_maps` results compose directly.
- **Masks/filters**: sphere, ellipsoid, box, cylinder, file; soft edges; band-pass in Å.

## 4. Dashboard (GUI)

A browser-based dashboard so the whole pipeline can be run without the command line:
`py3sta dashboard project/ [--port 8050]`. It runs on the GPU workstation; from a laptop use an SSH
tunnel (`ssh -L 8050:localhost:8050 workstation`).

**Principles**
- The project folder is the single source of truth. The dashboard reads and writes the same configs,
  STAR files and manifests as the CLI, so GUI and CLI can be mixed freely (start in the GUI, finish on
  the command line, or the reverse).
- No processing logic in the GUI: every button calls the same Python API the CLI uses.
- Long steps run as background jobs (one queue per GPU) with live logs and progress; the browser can be
  closed and reopened without killing jobs.

**Pages**
1. **Project** — create/open a project: input STAR, tomograms (`rlnMicrographName`), Warp XMLs
   (`add-xml`), box, pixel size, CTF mode (with a `ctf-check` button), GPU/CPU settings. Shows particle
   counts per tomogram, active vs excluded.
2. **Cycles overview** — one row per cycle with the state of each step (done / running / stale /
   missing), particle counts, FSC and resolution per cycle, global-average thumbnails; buttons for
   "run next step", "run to …", "re-run from …", "new cycle".
3. **Cycle parameters** — form editor for `cycle.yaml` (reference mode and masks, stack size, alignment
   search ranges, PCA factors, classifiers and K list) with validation, defaults and help text; "copy from
   previous cycle".
4. **Viewers**
   - volumes: orthogonal slices (XY/XZ/YZ) and projections with a slider, low-pass control; global
     average, references, class averages, eigenvolumes side by side; masks overlaid;
   - FSC curves (half maps, vs reference), scree plot, class counts, dendrogram;
   - "Open in ChimeraX" (REST remote control) for any map.
5. **Explore / classify** — the UMAP/PCA explorer (§2 step 7) embedded: colour by class / score /
   tomogram, **lasso to define groups**, live class average of the current lasso selection (GPU),
   compare K values and methods side by side.
6. **Select** — choose a classification (K, method, or lasso groups); per class a card with its average,
   member count and an `align / leave / exclude` toggle; target `each other` or a reference MRC; preview
   of the aligned class averages; "write _final.star".
7. **Particles** — table/filter of particles (tomogram, class history, scores, excluded flag); bulk
   include/exclude (`py3sta flag`); export STAR (RELION 4 / Warp).
8. **Jobs** — queue, running jobs, logs, GPU memory/utilisation, cancel/retry.

**Technology (default; revisit when Phase 6 starts)**
- Backend: FastAPI (REST + websocket for logs/progress) wrapping the py3sta API; job runner with
  subprocesses (one worker per GPU).
- Frontend: Plotly Dash for speed of development (pure Python, built-in lasso, sliders, forms); switch
  to a FastAPI + small JS frontend only if Dash becomes limiting.
- Images: slices rendered server-side (PNG) or sent as small arrays; volumes never shipped whole to the browser.
- Single-user, local; optional token for shared workstations.

**Phasing**
- v0 (with Phase 4): stand-alone explorer HTML (lasso → groups.json).
- v1 (Phase 6) ✅ 2026-10-16: read-only dashboard — cycles overview, viewers, FSC, logs, ChimeraX.
- v2 (Phase 6): run/edit — job queue, parameter forms, select page, flag management, live lasso averages.

## 5. Phases and acceptance checks

| Phase | Content | Acceptance |
|---|---|---|
| 0 ✅ | I/O, converters, extraction, wedge/dose/CTF averaging, parallel CPU, torch mirror | see PROGRESS.md |
| 1 | Pipeline skeleton: project/cycle dirs, configs, manifests, status, references (global/file), stack, pose composition, `_final.star`, `next`; dummy classification from existing `rlnClassNumber` | average from the stack == average from the tomograms (float16 vs float32 measured); only changed/drifted particles re-cropped; synthetic class transforms recovered exactly |
| 2 | Alignment engine (CPU + GPU), MRA step | MIM019 angles perturbed ±15° recovered (median < 5°), FSC not worse; 265 ptcl × 2000 orientations < 1 min on GPU |
| 3 ✅ | PCA (wedge-masked differences + plain), k-means, Ward, GMM, multi-K outputs | synthetic 2-state data separated; wedge factors absent |
| 4 ✅ | UMAP + `explore` (static HTML lasso → groups.json → class_lasso.star) | groups round-trip into `select`; lasso on synthetic data reproduces true classes |
| 5 | `select`: each/reference targets, leave/exclude (flag carried over), `flag --include/--exclude`, class-average alignment, propagate to members, full cycle | one full cycle on MIM019; next-cycle average ≥ previous (FSC, CC vs template) |
| 6 | Dashboard (§4): v1 read-only (cycles, viewers, FSC, logs), v2 run/edit (job queue, forms, select, flags, live lasso averages); ChimeraX push | a full cycle on MIM019 run entirely from the dashboard; CLI and GUI interchangeable on the same project |
| 7 | CTF in alignment/MSA scores, multi-tomogram datasets (height sign), novaCTF comparison | — |

## 6. Open questions (ask the user)

- ~~Reference for class-average alignment when `each`~~ — decided 2026-10-16: best-SNR class (`select.anchor` overrides).
- ~~Dashboard framework~~ — decided 2026-10-16: Dash; shared workstation (access token, one job queue per project).
