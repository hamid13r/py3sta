# py3sta — Progress

Update at the end of every work session: move items, add findings with numbers, note open questions.

## Current focus

**Phase 1 — pipeline skeleton** (see PLAN.md §1–2, §5).

Done:
1. ✅ `project.yaml` / `cycle.yaml` schema + loader with validation; `py3sta init` (2026-10-14, see below).

Next concrete tasks (revised 2026-10-15, see PLAN.md §1 for the decisions):
2. Manifests + `py3sta status` + `py3sta run`; CTF mode moves from `project.yaml` to `cycle.yaml`;
   `00_input.star` becomes a copy; project-level unrotated stack (create + incremental re-crop);
   `01_average` from the stack (one interpolation, per-tomogram CC).
3. Reference step (global, file) with masks + band-pass in Å.
4. Pose composition function + synthetic tests; `_final.star` writer; `py3sta next` (copies 00_input).
5. Dummy classification from existing `rlnClassNumber` to run select/final end-to-end before the real engines exist.
6. `py3staExcluded` column: honoured by every step from the start, `py3sta flag`.
7. `py3sta set-tomograms` STAR helper (swap reconstructions by `tomo_key()`).

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
