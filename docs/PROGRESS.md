# py3sta — Progress

Update at the end of every work session: move items, add findings with numbers, note open questions.

## Current focus

**Phase 1 — pipeline skeleton** (see PLAN.md §1–2, §5).

Done:
1. ✅ `project.yaml` / `cycle.yaml` schema + loader with validation; `py3sta init` (2026-10-14).
2. ✅ Manifests + `py3sta status` / `run` / `stack`; project-level unrotated float16 stack with incremental
   re-crop; `01_average` from the stack with per-tomogram CC; CTF mode in `cycle.yaml`; `00_input.star`
   is a copy (2026-10-15, see below).

3. ✅ Reference step (global, file, multi-reference) with masks + band-pass in Å; project-level
   `invert_contrast` (2026-10-15, see below).

4. ✅ Pose composition (`poses.py`), `08_final` → `cycle-NNN_final.star`, `py3sta next` (2026-10-15, see below).

5. ✅ `05_classify` (method `column`) + minimal `07_select` (align / leave / exclude, target reference or MRC):
   full cycle average → reference → classify → select → final → next runs end to end (2026-10-15, see below).

6. ✅ `py3sta flag` (2026-10-15, see below). Task 7 (`set-tomograms` helper) dropped by the user: tomogram
   paths live in the STAR file and are edited there.

**Phase 2 — particle alignment (MRA)**, in three parts:
1. ✅ Scoring engine `mra.py` (numpy reference + torch batches, cone/spin levels, constrained CC) (2026-10-15).
2. ✅ `03_align` step (2026-10-15, see below).
3. MIM019 checks (2026-10-15, see below): timing ✅ (44 s), recovery ✅ (median 2.07°, same as a control run
   from the true poses: 2.06°); FSC equal to the control, 0.6–0.9 Å short of RELION's poses — see "Phase 2 part 3".

## Phase 2 part 3 — MIM019 alignment checks (2026-10-15, RTX A5000)

- Speed: crops band-passed on the GPU (was numpy, ~25 % of the time) → **11 000 evaluations/s** at bin 2
  (box 72 → 36, constrained CC), ~1 500/s at bin 1. Batch 256 / chunk 32 defaults (larger batches no faster).
- **Timing: 265 particles (MIM019_2_lam11_ts_003) × 1 980 orientations, bin 2: 44 s** (target < 1 min ✅).
- Bug found: the total shift was clipped per axis in the reference frame, but the crop allowance is per
  tomogram axis (a ±2 cube reaches 2√3 along a tomogram axis) → 58 particles left their crop. Now clipped per
  tomogram axis (`max_shift` is per tomogram axis); test added.
- Recovery: all 6081 particles perturbed by a random rotation (uniform 0–15°, median 7.5°), aligned to the
  cycle-0 average (sphere mask 191.5 Å, low-pass 30 Å), levels `[[18,6,18,6],[6,2,6,2],[2,0.5,2,0.5]]`, bin 2,
  max shift 2 voxels:

| | FSC 0.5 | FSC 0.143 | CC with original average | error vs RELION poses |
|---|---|---|---|---|
| original (RELION) poses | 24.9 Å | 16.3 Å | 1.0000 | — |
| perturbed | 34.5 Å | 22.5 Å | 0.9696 | median 7.5° |
| aligned, bin 2 (580 s) | 25.6 Å | 16.8 Å | 0.9977 | **median 2.07°**, 90 % 8.9°; centre 3.35 Å |
| + bin 1 rotation-only refinement (1 589 s, low-pass 20 Å) | 25.7 Å | 17.9 Å | 0.9977 | median 2.08°, 90 % 8.6° |

- 782 of 6081 (13 %) at the search edge after bin 2 (mostly the shift limit; median CC 0.36 — low-SNR particles).
- The bin-1 pass did not help (FSC 0.143 got worse: refining against a single full-map reference at 20 Å
  probably fits noise; references are not half-set, by decision). A second pass with shifts needs re-cropped
  particles (1486 had used their crop allowance) — the pipeline does that between cycles.
- **Control** (same settings, starting from the *unperturbed* RELION poses): the poses move by the same
  amount — median 2.06°, 90 % 8.5°, centre 3.35 Å — and the FSC is the same (25.8 / 16.9 Å; CC with the
  original average 0.9977). So the perturbed run converged to the same state as a run started from the
  truth: ~2° is the precision of this search (bin 2, 30 Å, one full-map reference) on this data, not a
  recovery failure. The FSC loss vs RELION (0.6–0.9 Å) is what one pass at these settings costs relative to
  RELION's high-resolution refinement.
  - Change distribution: 913 < 1°, 2006 1–2°, 1967 2–5°, 684 5–10°, 511 > 10°.
  - Low-SNR particles move most: median change 5.21° in the lowest CC quarter vs 1.17° in the highest.
  - Acceptance "FSC not worse": met relative to a run from the true poses; not met relative to RELION.
    Options: finer final sampling with a matching reference band, restraining large changes for low-CC
    particles (i3 practice: small cones), or half-set references (decided against so far).

## Phase 2 part 2 — `03_align` step (2026-10-15)

- `align:` in `cycle.yaml`: `enable` (false = poses passed through), `levels`, `max_shift` (Å; null = 2 voxels),
  `lowpass`, `highpass`, `bin`, `wedge` (constrained CC; needs `wedge.use`), `mask` (null = the 02_reference
  mask). Validated: levels shape, binned Nyquist, box divisible by bin, shift within the stack allowance.
- References = `02_reference/references.mrcs` (all of them: multi-reference). Outputs: `aligned.star` (every
  row of the pose source; active rows get composed poses, `rlnBestReference`, `py3staRefCC`,
  `py3staAlignAngle`, `py3staAlignShift` (Å), `py3staAlignEdge`; inactive rows 0), `align.csv` (per particle:
  best reference, CC per reference, angle change, shift xyz Å, edge flag); manifest info with medians,
  edge count (warning above 10 %), best-reference counts. classify / select / final now start from these poses.
- **Shift allowance = stack.margin − 2 voxels** (not − 1): a freshly cut crop may already be 0.5 voxel
  off-centre, so `margin − 1` left no room and every particle failed the drift check. `run` refreshes the stack
  with this cycle's shift limit first (particles that could leave their crop are re-cropped). MIM019
  (margin 4, 6.65 Å): up to 2 voxels = 13.3 Å per cycle.
- `update_particles` helper: writes per-particle columns by `rlnTomoParticleName`; new columns default to 0
  (no NaN in STAR files).
- Tests (`test_pipeline_align.py`, chiral object, 16 particles, poses off by ±5° per angle and ±0.8 voxel,
  torch on CPU): median error 0.41°, one particle 2.2° — it scores higher there than at its true pose (0.9640
  vs 0.9624), so a limit of the score on this smooth noisy object, not a search miss; excluded rows untouched;
  `rlnClassNumber` not written; downstream steps use the aligned poses; changing `align.*` makes classify
  stale; multi-reference (mirror vs true object) → `rlnBestReference` = 2 for all; config errors.

## Phase 2 part 1 — particle alignment engine, `mra.py` (2026-10-15)

- Scoring in the reference frame: trial pose M' = D·M, particle sampled from its unrotated stack crop at M'
  (one interpolation), masked Pearson CC with the reference over all shifts (align.py's Fixed / ncc_map
  core); result (D, t) composes with poses.py. Band-pass applied once per crop; `bin` samples every b-th
  voxel of the (low-passed) crop and Fourier-resamples references + mask.
- Search: i3 levels `[nut, nstep, spin, sstep]` (`cone_grid`: z axis tilted ≤ nut, spin ±spin), coarse to
  fine around the current pose; each reference searched separately, best CC wins (`best_ref`, `cc_ref`).
  Flags particles whose best pose is on the border of the first level or at the shift limit (`at_edge`).
- Two findings while testing on the chiral object (wedge ±50°, poses off by up to 15° and 1.5 voxels):
  1. **Integer shifts bias the rotation**: a few degrees of rotation mimic half a voxel of shift, so the
     search preferred wrong rotations (CC at truth 0.967 < found 0.972). Fix: after every level the particle
     is re-centred on the sub-voxel peak (parabola), and the final shift is converged at the best rotation.
  2. **Filtering only the reference with the wedge made the score erratic** (±2° around the truth scored
     0.974 / 0.976 non-monotonically). Fix: the sampled particle gets the same rotated coverage filter, so
     both maps share exactly the same Fourier support → clean peak at the truth.
  3. Level coverage: each level's range must cover the previous step (5° → ±2° left gaps; 5° → 2° → 0.5° works).
     Default levels: `[[12, 4, 12, 4], [3, 1, 3, 1], [1, 0.5, 1, 0.5]]`.
- Tests (`test_mra.py`): cone grid geometry; torch == numpy scores (2e-4, identical shifts; wedge on/off,
  bin 1/2); numpy driver == torch driver; recovery; constrained beats plain; multi-reference (object vs
  its mirror: every particle picks its own); edge flag.

| synthetic recovery (8 particles, wedge ±50°, perturbation median 6.9°, max 15.3°) | angle error median / max | shift max |
|---|---|---|
| constrained CC | **0.83° / 1.00°** | 0.07 voxel |
| plain CC | 2.14° / 3.56° | 0.10 voxel |

## Phase 1 task 6 — `py3sta flag` (2026-10-15)

- `pipeline/flags.py` (`flag_particles`, `select_particles`), CLI `py3sta flag PROJECT CYCLE --exclude SEL
  --include SEL [--dry-run]`. Edits the cycle's `00_input.star` in place (tracked as a STAR edit: the cycle's
  steps become stale with "00_input.star changed"); one line per change in `cycle-NNN/flag.log`.
- Selections: `class:CYCLE:K[,K]` (final STAR of that cycle, else `selected.star`), `tomo:NAME[,NAME]`,
  `excluded:CYCLE` (`py3staExcludedCycle`), `file:PATH` (STAR with `rlnTomoParticleName` or one name per line),
  `all`. Unknown classes / tomograms / names, a particle in both lists, or a running step are errors.
- exclude sets `py3staExcluded = 1`, `py3staExcludedCycle = CYCLE`; include sets 0 and keeps the cycle (history).

| MIM019 check (`data/mim019_select_check` cycle 1; exclude the 3 low-CC tomograms) | all | 3 excluded |
|---|---|---|
| particles used | 6002 | 5691 (313 flagged) |
| FSC 0.5 | 24.9 Å | 24.8 Å |
| FSC 0.143 (first crossing) | 16.3 Å | 16.3 Å |
| FSC 30–18 Å | — | +0.001 to +0.014 |
| per-tomogram CC median / flagged | 0.947 / 3 | 0.949 / none |

The "last 0.143 crossing" changed 13.3 → 16.3 Å only because the FSC hovers at the threshold (0.146 vs 0.141
at 13.3 Å): report the first crossing.

## Phase 1 task 5 — `05_classify` (column) + minimal `07_select` (2026-10-15)

- `classify.method: column` takes classes from any integer column (`classify.column`, default
  `rlnClassNumber`; missing column = one class, with a warning; non-integer values are an error). No upstream
  except the pose source, so it runs before MSA exists. Outputs `class_column.star` (particle → `py3staClass`),
  `classes.mrcs` (class averages from the stack with the cycle's averaging settings), `classes.star` (counts).
- `select:` section of `cycle.yaml`: `align` (`all` = every class not listed elsewhere, or a list), `leave`,
  `exclude`, `target` (`reference` = 02_reference reference 1, or an MRC path, resampled), search settings
  (`lowpass`, `highpass`, `bin`, `step`, `max_shift`; validated against the binned Nyquist). Each aligned class
  average goes through `align_maps` (no mirror: a mirrored transform cannot be applied to poses) with the
  02_reference mask; the transform is composed onto its members (`apply_delta`). Outputs `selected.star` (every
  row of the pose source; `rlnClassNumber`, `py3staClassCC`, exclusion with `py3staExcludedCycle`),
  `transforms.star` (per class: action, angles, shift Å, CC), `classes_aligned.mrcs`.
- Pose sources generalised: classify, select and final each start from the poses of the last implemented
  pose-writing step before them (align → select) and record it as upstream; `load_cycle_input(path=)`.
- `_averager` helper shared by 01_average and 05_classify (same wedge / CTF / normalisation / device).
- Tests (`test_pipeline_select.py`, chiral Gaussian object placed analytically in 2 tomograms, box 40): class 2
  poses perturbed by a known (R, t) → after classify + select all poses within 1° / 0.3 voxel of the truth;
  leave keeps poses, exclude flags members and the next cycle skips them; staleness (column change → classify
  stale → select → final), edited `selected.star` protected, config validation, missing/non-integer column.

| MIM019 check (6081 particles, classes = rlnRandomSubset, class 2 perturbed by R = (20°, 15°, −10°), t = (2, 0, −1) voxels; target = unperturbed cycle-0 average) | value |
|---|---|
| class 2 transform found | (20.0°, 15.0°, −10.0°), shift (13.3, 0, −6.6) Å — exact; CC 0.999 |
| class 1 transform found | identity (tilt 0, rot = −psi), shift 0; CC 0.999 |
| final poses vs truth (mean) | class 1 0.04° / 0.04 Å, class 2 0.02° / 0.03 Å |
| cycle-1 average vs original cycle-0 average | CC **1.0000** (perturbed cycle-0 average: 0.840) |
| time, cycle 0 on RTX A5000 | 33 s (average 6.5, classify 5.7, select 19.4 — global 10° search per class) |
| stack | untouched (no drift beyond the margin) |

Check project: `data/mim019_select_check/` (stack symlinked to `data/mim019_project/stack`).
The pose source is a step parameter: `data/mim019_project/cycle-000` now shows classify/select missing and
`08_final stale: pose_source <unset>→select`.

## Phase 1 task 4 — pose composition, `08_final`, `py3sta next` (2026-10-15)

- `poses.py`: a delta (R, t) in the reference frame (t in box voxels) means "new subvolume = old one sampled
  at Rᵀ(r + t)" — the same meaning as `align.apply_alignment`, so `align_maps` results compose directly:
  **M' = R·M, c' = c + M'ᵀ·t**. `compose`, `chain` ((R2R1, R2t1 + t2)), `invert` ((Rᵀ, −Rᵀt)),
  `apply_delta(ps, R, t, apix, rows=)` (one delta for a class or one per particle; new angles; shift goes into
  `rlnOrigin*Angst`, coordinates unchanged as in RELION). Mirrored transforms are refused.
- Bug fixed in `geometry.matrix_to_euler`: within ~1e-6 of gimbal lock (tilt ≈ 0/180°) rot and psi were each
  computed from tiny numbers and no longer reproduced the matrix (error 0.05 seen in the round trip).
  psi is now taken from the residual rotation Ry(−tilt)·Rz(−rot)·M, so the angles always reproduce M.
- `08_final` (step `final`): poses from the last implemented step that writes poses (`07_select/selected.star`,
  else `03_align/aligned.star`, else `00_input.star`; recorded as upstream), every row kept (excluded too),
  `py3staCycle` added, all STAR blocks kept, refuses if the pose source lost particles. Output is a tracked
  (editable) STAR. `rlnClassNumber` comes later with select.
- `py3sta next PROJECT CYCLE` (`project.next_cycle`): new `cycle-NNN+1/` with `cycle.yaml` copied (only the
  `cycle:` line changed, comments kept) and `00_input.star` = copy of the final STAR (hand edits included,
  with a warning). Refuses if `final` is missing/stale or the next cycle exists (`--force`).
- Tests: chain/invert identities; round trip to 1e-10 (matrices) / 1e-6 px (centres) incl. gimbal lock and
  per-particle deltas; extraction with the composed pose == obj(Rᵀ(r + t)) (CC > 0.995, analytic tomogram,
  ts_apix ≠ apix); two classes with different known errors: `align_maps` on each class average + composition
  restores all member poses (< 1°, < 0.2 voxel); final/next/staleness/edit/pose-source pipeline tests.

| MIM019 check (6081 particles, torch cuda:0) | value |
|---|---|
| average with poses composed with R = (25°, 10°, −15°), t = (2, 0, −1) voxels vs old average moved by (R, t) | CC **0.9993** |
| same vs old average unmoved (control) | CC 0.496 |
| round trip (delta then inverse) | matrices 1.5e-15, centres 5.7e-14 px |
| average time from the stack (original / composed poses) | 5.6 s / 5.3 s |
| `run --from final` + `next` on `data/mim019_project` | 0.2 s; `cycle-001/00_input.star` byte-identical to the final STAR |

## Phase 2 (started early) — map-to-map alignment, `py3sta align-maps` (2026-10-15)

- `align.py`: `resample` (Fourier crop/pad, centre voxel kept, input padded so the output voxel size is
  exact to ~1e-4), `fourier_shift`, `center_of_mass`, `mirror` (about voxel n//2), `load_map`,
  `so3_grid` (10°: 14 832 rotations), `local_grid`, masked Pearson CC over all shifts by FFT (`ncc_map`,
  numpy; batched torch version, parity test), `align_maps` (global grid on binned/low-passed maps → local
  grids around the top 3 → local grid at full sampling → polish; both hands), `apply_alignment`,
  `align_map_files`. Convention: the result is the pose (rlnAngle*, shift in the fixed frame) of the fixed
  structure inside the moving map, i.e. aligned = `extract(moving, c + M.T t, M)` — same as particles.
- **Polish finding:** RealSpaceTM's polish optimises the rotation against the best *integer-shift* CC. With a
  fractional true shift this is biased: a 2–4° rotation error mimics a half-voxel shift and scores higher
  (synthetic: true pose 0.980 vs 3° off 0.990). py3sta polishes rotation + sub-voxel shift together
  (6-D Nelder-Mead) on the CC of the map extracted at exactly that pose; shifts bounded by `--max-shift`.
  Synthetic accuracy: < 0.15° and < 0.005 voxel (4 random poses), 2.6 s on GPU (40³), 29 s numpy.
- MRC reader handles any MAPC/R/S axis order (EMD-30432 is 2,1,3). Reference files with another voxel
  size are now resampled (`reference.load_volumes`), so EMDB maps work as `file` references.
- New core dependency: scipy (Nelder-Mead).
- CLI `py3sta align-maps FIXED MOVING... -o DIR` (mask, lowpass, bin, step, max-shift, polish, no-mirror,
  invert, center, backend/gpu) → `<name>_aligned.mrc`, `<name>_fsc.txt`, `alignment.star`.

**Real data** (fixed = MIM019 cycle-000 average, white, 72³ @ 6.65 Å; RTX A5000; ~30 s per map incl. both hands):

| Moving map | Result |
|---|---|
| user's `average.mrc` with a known pose (−48, 117, 63)°, shift (17.29, −11.30, 21.28) Å, `--invert` | recovered (−48.00, 117.00, 62.99)°, (17.29, −11.30, 21.29) Å; CC 0.997 (mirror 0.486) |
| same without `--invert` | CC 0.08, warning "contrast inverted?" (min CC −0.88) |
| EMD-30432 (mouse kidney 80S, 2.82 Å, 400³ @ 1.04 Å) | direct hand CC 0.787 vs mirror 0.527 → **tomogram hand matches EMDB**; FSC 0.5 at 22 Å; pose (−20.7, 125.6, −116.6)°, shift ~8 Å |
| IsoNet average, `--invert` | identity rotation, zero shift, CC 0.943 (same poses, as expected) |

Files: `data/align_test/` (B/compare_emdb.png shows the fit). EMDB map: `data/emdb/`.

## Phase 1 task 3 — `02_reference`, contrast (2026-10-15)

- `reference.py`: `make_mask` (sphere / ellipsoid / box / file; raised-cosine edge centred on the size, Å;
  centre offset in Å), `bandpass` (raised-cosine edges, 3 Fourier shells), `fit_box` (pad/crop keeping
  voxel n//2 → box//2), `prepare_reference` (invert → band-pass → normalise inside mask → mask),
  `load_volumes` (MRC or MRC stack; voxel size must match, no resampling), `masked_cc`.
- Step `02_reference`: upstream = `average` only in `global` mode (`Step.upstream` may depend on the
  cycle config); `file` mode runs without an average. Outputs `references.mrcs` (float32), `references.star`
  (source, file, invert, CC with the current average), `mask.mrc`. Negative CC → "contrast inverted?" warning.
  `classes` mode refused until `select` exists. All references share one mask (as i3).
- `invert_contrast` (project.yaml; `py3sta average --invert-contrast`): densities × −1 on read, both paths,
  stack unchanged. `01_average` records `protein_contrast` (inside particle radius vs outer shell) and warns
  when protein is dark.
- Tests: 78 pass (`test_reference.py`: masks, band-pass, fit_box parity cases, MRC stacks, contrast,
  step staleness, file mode without average, inverted-file warning, replaced file, invert_contrast).
- MIM019: raw average reads `dark` (−1.3), with `invert_contrast: true` `light` (+1.4; project now uses it).
  Global reference CC 1.000; user's `average.mrc` (box 64 → 72) CC −0.999 without `reference.invert`
  (warned), 0.999 with; two-reference stack (user average + IsoNet average) CC 0.999 / 0.944.

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

- (2026-10-15) **IsoNet-corrected tomograms** (`isonet_denoising/corrected_tomos`, isonet2 n2n unet-medium), 6002
  particles, no wedge weighting (IsoNet fills the wedge): the map looks smoother and less detailed than raw.
  Half-map FSC rises again above 0.045 1/Å (0.69 at 15 Å, 0.8 near Nyquist; raw falls to ~0.15): the
  network output is correlated between half sets, so **FSC is not a valid resolution measure on IsoNet tomograms**.
  CC with the raw-tomogram average 0.946. Files: `data/isonet_average/`.

## Decisions

- (2026-10-15) Multi-reference alignment writes the winning reference to **`rlnBestReference`** (user's
  choice of name; not a RELION label) and its CC to `py3staRefCC`; `rlnClassNumber` stays reserved for
  classification.
- (2026-10-15) Class selection lives in the `select:` section of `cycle.yaml` (tracked like every parameter);
  the planned `py3sta select` CLI / dashboard will only write that section. `target: each` comes in Phase 5.
- (2026-10-15) No automatic fitting of `file` references (no `reference.fit`): the user aligns external maps
  with `py3sta align-maps` and provides the fitted map — routine practice.

- (2026-10-15) No content hashing of large files: run ids between steps, size/mtime for external files.
  STAR files stay user-editable: fingerprinted (hash only when size/mtime changed), edits tracked, never lost.
- (2026-10-15) No tomogram sets: tomogram paths live only in the STAR file (edit `00_input.star` to switch
  reconstructions); the CTF mode moves to `cycle.yaml`. `00_input.star` is a copy, not a link.
- (2026-10-15) Running a step with a stale/missing/failed upstream is refused unless `--force`.
- (2026-10-15) References are full maps (no half-set references). Multi-reference alignment (Phase 2): every
  particle against all references, best CC kept (index → `rlnClassNumber`).
- (2026-10-15) White-on-black via project-level `invert_contrast` (applied on read; stack keeps raw values).
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
