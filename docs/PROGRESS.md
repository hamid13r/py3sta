# py3sta — Progress

Update at the end of every work session: move items, add findings with numbers, note open questions.

## Current focus

**Phase 1 — pipeline skeleton** (see PLAN.md §1–2, §5).

Done:
1. ✅ `project.yaml` / `cycle.yaml` schema + loader with validation; `py3sta init` (2026-10-14).
2. ✅ Manifests + `py3sta status` / `run` / `stack`; project-level unrotated float16 stack with incremental
   re-crop; `01_average` from the stack with per-tomogram CC; CTF mode in `cycle.yaml`; `00_input.star`
   is a copy (2026-10-15, see below).

Next concrete tasks:
3. Reference step (global, file) with masks + band-pass in Å.
4. Pose composition function + synthetic tests; `_final.star` writer; `py3sta next` (copies 00_input).
5. Dummy classification from existing `rlnClassNumber` to run select/final end-to-end before the real engines exist.
6. `py3sta flag` (the `py3staExcluded` filter itself is already honoured by `load_cycle_input`).
7. `py3sta set-tomograms` STAR helper (swap reconstructions by `tomo_key()`).

## Phase 1 task 2 — manifests, stack, `01_average` (2026-10-15)

- `stack.py`: `refresh_stack` / `ParticleStack` (`stack/stack.mrcs` float16, `stack.star`). Crops of
  box + 2·margin on whole voxels around the rounded centre; `max_drift(margin) = margin − 1` voxels per
  axis keeps the extraction sphere + 1 voxel inside the crop (tested for boxes 16/15/12 at the limit:
  identical to sampling the tomogram). Re-crop on tomogram path/size/mtime change or drift; new
  particles appended; rows marked invalid before writing (interrupted refresh is repaired next time).
  Many particles per tomogram → one sequential z-slab read; ≤ 8 workers.
- `average.py` / `backend_torch.py`: subvolumes are defined inside the **extraction sphere** (radius
  box//2 − 1) in both the tomogram and the stack path, so the two agree; **normalisation per particle
  inside `particle_radius`** (project.yaml, Å, default half the box), separate from step masks.
  `extract_crop`, `sample_cubes` (torch), `average_particles(..., stack=, per_tomo=)`, `per_tomo_cc`.
  Torch tomogram path now cuts only box+3 cubes (100 ptcl: 2.5 s → 0.3 s).
- Per-tomogram check: each tomogram's average vs the others' (leave-one-out, CC inside the particle
  radius after 40 Å low-pass). CC grows with particle count, so tomograms are compared by the implied
  per-particle SNR cc²/(n(1−cc²)); flagged if the robust z of log(SNR) < −3. (A plain CC rule flagged
  5 tomograms on MIM019, incl. ones with 6 and 10 particles; the SNR rule flags 3.)
- `pipeline/manifest.py`: `run_step` / `check_state` (fingerprints, size/mtime, run ids, edited-output
  protection with backup, failed / interrupted, input changed during run). `pipeline/steps.py` (step
  order, params, `01_average`), `pipeline/runner.py` (`run_cycle` with option (a), `refresh_project_stack`,
  `project_status`). CLI: `run`, `status`, `stack`; `init --particle-radius --margin`; `average --particle-radius`.
- Tests: 62 pass (`test_stack.py`, `test_pipeline_run.py`, updated `test_pipeline_config.py`).

**MIM019 real data** (`data/mim019_project`, 6081 particles, 34 tomograms, box 72, margin 4, RTX A5000):

| Check | Result |
|---|---|
| stack build (34 × 2.9 GB tomograms) | 103 s, 6.2 GB, 7 GB peak RAM |
| `01_average` from the float16 stack (torch) | 6.4 s (from tomograms: 10.9 s); 6002 used, 79 near edges |
| stack float16 vs tomogram average (in sphere) | max diff 7.4e-5 of max; float32 stack 7.5e-5 → **float16 adds no error** |
| FSC 0.143 (unmasked halves) | first crossing 16.3 Å (last 13.3 Å = Nyquist, tail hovers at ~0.14) — same for all three |
| CC with user's `py3sta_test/average.mrc` (64³ centre, sphere r 31) | **0.9988**, no shift |
| per-tomogram flags | `2_lam7_ts_007` (z −6.4), `3_lam4_ts_003` (−4.5), `3_lam5_ts_007` (−3.6, 10 ptcl) — worth a look |
| `00_input.star` edit pointing `2_lam1_ts_002` to another tomogram | status: stale (00_input changed); 196 rows re-cropped (28 s); that tomogram CC 0.95 → 0.03, flagged (z −31) |
| 10 particles shifted by 6 voxels | status: "10 particle(s) beyond half the drift allowance"; run re-crops exactly 10 (0.2 s) |

## Done (Phase 0)

| Area | What | Where |
|---|---|---|
| I/O | RELION 4 STAR read/write (no deps), MRC read/write (memmap), i3 `.trf`, dipole `.pos` | `io/` |
| Converters | `pos2star`, `trf2star`, `star2trf`, `star2warp` (replace `pos2i3.py`, `trf2star.py`) | `cli.py` |
| Geometry | RELION ZYZ conventions, verified 4 ways (see CLAUDE.md) | `geometry.py` |
| Extraction | trilinear (flat `take`, ~4.5x faster than fancy indexing), memmapped crops | `average.py` |
| Averaging | real-space and Fourier Wiener; half maps; FSC (first/last crossing) | `average.py` |
| Parallel | `-j N` worker processes; 263 ptcl: 24 s → 5.5 s on 4 cores | `average.py` |
| GPU | torch mirror of extraction/wedge/dose/CTF/average, `--backend torch --gpu N` — **runs on the user's GPU (CUDA 12.2)** | `backend_torch.py`, `scripts/check_torch.py` |
| Wedge | Warp XML / .tlt; sign measured from data (`wedge-check`) | `wedge.py` |
| Dose | Grant & Grigorieff exposure filter per nearest tilt | `wedge.py` |
| CTF | per-tilt GridCTF defocus + particle height; flip / wiener / amplitude; `ctf-check` | `wedge.py`, `cli.py` |
| Files from STAR | tomograms from `rlnMicrographName`; tilts/CTF from `rlnWarpXMLfile` (`add-xml` utility) | `average.py`, `utils/add_warp_xml.py` |
| Tests | 25 numpy tests pass; 5 skip without scipy/torch (run locally) | `tests/` |

## Findings (MIM019_2_lam11_ts_003, 265 particles, 6.65 Å, box 72)

- Global average matches `template.mrc`: CC 0.73 (inverted template); unrotated control 0.57; mirrored 0.42.
- Wedge sign for Warp XML (`AreAnglesInverted=True`) is −1: coverage −67.6…+54.6°.
- Wedge weighting did not change FSC here (orientations diverse; coverage ≥ 61 % everywhere).
- Half-map FSC dips at CTF zeros 37 / 26 / 21 Å (defocus ~6.5 µm). Report last 0.143 crossing (~22–28 Å).
- **Warp `ts_reconstruct` and novaCTF tomograms are already phase-flipped**; etomo WBP is not
  (correlation with CTF-corrected template between the first two zeros: Warp +0.60 / flipped −0.58;
  etomo −0.53 / flipped +0.57; novaCTF +0.66 / −0.61). Half-map FSC cannot detect this.
- novaCTF DW tomogram gives the best average without extra correction (mean half-FSC 15–40 Å 0.169 vs Warp 0.148).
- CTF height sign (`--ctf-height-sign`) not resolvable on one tomogram; test on multi-tomogram data.
- BLAS matmul inside worker processes oversubscribed the CPU (−j4 slower than −j1); fixed with element-wise math.

## Decisions

- (2026-10-15) No content hashing of large files: run ids between steps, size/mtime for external files.
  STAR files stay user-editable: fingerprinted (hash only when size/mtime changed), edits tracked, never lost.
- (2026-10-15) No tomogram sets: tomogram paths live only in the STAR file (edit `00_input.star` to switch
  reconstructions); the CTF mode moves to `cycle.yaml`. `00_input.star` is a copy, not a link.
- (2026-10-15) Running a step with a stale/missing/failed upstream is refused unless `--force`.
- (2026-10-15) Normalisation per particle inside `particle_radius` (Å, project.yaml; default half the box),
  separate from the masks of later steps. Subvolumes are zero outside the extraction sphere (box//2 − 1).
- (2026-10-15) Stack = raw **unrotated** float16 crops (box + 2·margin, margin 4 voxels), project-level,
  shared by all cycles; the pose is applied by each consumer → one interpolation in the whole pipeline.
  Re-crop only on tomogram change or drift beyond the margin; alignment shift limit ≤ margin.
  Step order: average → reference → align → msa → classify → embed → select.

- Package/CLI name `py3sta` (GitHub `hamid13r/py3sta`); renaming later is cheap (SAGA was the favourite alternative).
- RELION 4 particle format; Warp export via minimal STAR (`star2warp`).
- Tomogram path in `rlnMicrographName`, Warp XML in `rlnWarpXMLfile`; no `--tomo-dir`.
- User wants PCA, k-means and UMAP, with lasso drawing on the UMAP plot to define particle groups (Phase 3–4).
- Excluded particles are **kept** in the STAR file with `py3staExcluded = 1`, carried over to all later
  cycles and skipped by every step, until the user flips the flag back (`py3sta flag --include`). (PLAN.md §2)
- A browser dashboard (GUI) for the whole pipeline is part of the plan (PLAN.md §4, Phase 6); CLI and GUI share the project folder.
- GPU backend confirmed working by the user with `py3sta average --backend torch` (CUDA 12.2).

## Phase 1 task 1 — configs + `py3sta init` (2026-10-14)

- `src/py3sta/pipeline/config.py`: dataclasses `ProjectConfig` (box, apix, ts_apix, wedge, ctf, compute)
  and `CycleConfig` (average, reference + mask, stack); strict YAML loader (unknown key → error with
  "did you mean"); all value errors collected and reported with file + dotted key; cycle checks against
  the project (mask radius + soft + |center| ≤ half box, lowpass ≥ Nyquist, msa_box ≤ box).
  No pydantic; new dependency `pyyaml`.
- `src/py3sta/pipeline/project.py`: `init_project()`, `load_project()`, `Project` (cycles, cycle_config).
  `init` checks that all tomograms (`rlnMicrographName`) and XMLs exist, that all tomograms share one
  voxel size (header rounded to 1e-4 Å, as MRC stores float32), and that the XMLs have CTF if `--ctf` is set.
  It writes `input.star` from the **raw STAR blocks** (all columns kept; only the path columns made
  absolute, keeping symlink names because `tomo_key()` reads the file name). `ParticleSet.to_relion4`
  is not used because it rewrites `rlnMicrographName` to `.tomostar`. `cycle-000/00_input.star` is a relative
  symlink to `../input.star`. Defaults: box = `rlnImageSize`, wedge on if `rlnWarpXMLfile` is present,
  mask radius 40 % / soft edge 10 % of the box, lowpass max(30 Å, 4·apix).
- Tests: `tests/test_pipeline_config.py` (12 tests: roundtrip, unknown keys, types, value checks,
  init on synthetic tomograms + XML, force, no-XML, mixed voxel size, missing tomogram, CLI).
- Real data: `py3sta init data/mim019_project --star .../py3sta_test/ribosome.star` → 6081 particles,
  34 tomograms, box 72 at 6.65 Å (478.8 Å), wedge on, tilt ranges ≈ −67…+47–56°, rlnRandomSubset
  2980/3101; 1.8 s. The project reloads and its `input.star` resolves all 34 tomograms.

## GPU parity (2026-10-14, `scripts/check_torch.py`, RTX A5000, torch 2.5.1+cu121)

Data: `/data1/users/hamid/robpicker_MIM019/py3sta_test/ribosome.star`, tomogram `MIM019_2_lam1_ts_002`
(196 rows, Warp XML tilts), box 72, 6.65 Å. Max relative difference torch vs numpy:

| Check | n=100 | n=196 |
|---|---|---|
| extraction (8 ptcl) | 1.1e-4 | — |
| wedge weights, no dose | 4.2e-5 | — |
| wedge weights, dose | 5.1e-3 (*) | — |
| average plain | 7.1e-6 — numpy 8.3 s / torch 2.5 s | 4.7e-6 — numpy 13.6 s / torch 2.1 s (6.5×) |
| average wedge | 4.9e-6 — numpy 7.8 s / torch 2.2 s | 3.4e-6 — numpy 15.2 s / torch 1.2 s (13×) |

(*) A single voxel in one particle (0.0607 vs 0.0658), where the nearest tilt switches because of float32
rounding. All other voxels agree to ~1e-5. Harmless.
numpy timings are single-process (`jobs=1`); torch time includes the first CUDA call.

Environment: `conda activate tomo_toolshed`. Its torch was `2.14.1+cu130`, which cannot use the CUDA 12.2
driver. With the user's OK (only py3sta uses torch there) it was replaced by `torch==2.5.1+cu121`
(`pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`); CUDA works, and `pip check`
reports no torch conflicts. A dedicated py3sta conda env is planned (user).

`test_io.py::test_dipoles_match_legacy_pos2i3` differed from scipy 1.15.3 `align_vectors` by up to 2.5e-5
(first run with scipy; non-normalised secondary vectors). Tolerance loosened to 1e-3 (user: dipole
conversion need not be precise). All tests pass.

## Open questions

- Dedicated py3sta conda env (user will create it); then update CLAUDE.md.
