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

**Phase 3 — MSA + classification** ✅ (2026-10-16, see below): `04_msa` (wmd / plain ± own wedge, randomized PCA,
wedge diagnostics, PNGs), `05_classify` (kmeans / ward / gmm, list of K, stability, `classify.use`), and the
on-demand `py3sta variance`.

**Phase 4 — embedding + lasso selection** ✅ (2026-10-16, see below): `06_embed`, `py3sta explore` (offline HTML,
lasso groups), groups as a classify run. Wedge factors: whitening per run (default), warnings only (2026-10-16, see below).

**Phase 5 — select + full cycle** (2026-10-16, see below): `select.target: each` (anchor = best-SNR class or
`select.anchor`) ✅; full MIM019 cycle 0 → 1 run end to end ✅. Acceptance mixed: with all classes aligned
cycle 1 is 0.5 Å worse (16.8 vs 16.3 Å) — the loss is 03_align from RELION's poses (same as Phase 2 part 3),
select is ~identity; excluding the 35 Å class keeps 16.3 Å with 60 % of the particles. Open, and now the main
issue: py3sta's alignment (bin 2, 30 Å) degrades RELION's poses at 15–25 Å (template FSC 0.83 → 0.64 at 19 Å;
see "Gold standard" below). `gold_standard: true` ✅ (2026-10-17). novaCTF vs Warp and `ctf: amplitude` checked.

**Phase 6 — dashboard** (started 2026-10-16, see below; user: Dash, shared workstation → token). v1 read-only ✅:
`py3sta dashboard` (cycles overview, Maps, Plots, Logs, ChimeraX). v2 (job queue, cycle.yaml form, select
page, explore + live lasso averages, particles / flags) on hold by the user.

**Phase 2 — particle alignment (MRA)**, in three parts:
1. ✅ Scoring engine `mra.py` (numpy reference + torch batches, cone/spin levels, constrained CC) (2026-10-15).
2. ✅ `03_align` step (2026-10-15, see below).
3. MIM019 checks (2026-10-15, see below): timing ✅ (44 s), recovery ✅ (median 2.07°, same as a control run
   from the true poses: 2.06°); FSC equal to the control, 0.6–0.9 Å short of RELION's poses — see "Phase 2 part 3".

## Phase 3 — MSA + classification (2026-10-16, RTX A5000)

Steps (one commit each): `msa.py` engine → `04_msa` step → `cluster.py` → multi-K `05_classify` → MIM019 checks.

- **MSA stack** (`msa.ParticleVolumes`): each particle sampled once at its current pose from the unrotated crop
  (every `bin`-th voxel of the band-passed crop, as in MRA), normalised inside the normalisation mask.
  Modes: `wmd` = W_i (F_i − A), A = Wiener average of the MSA stack itself (default); `plain` = x_i
  (`wedge: true` = W_i x_i, the user's choice for "plain with the wedge filter": own wedge on the particle, plain mean).
- **PCA**: randomized SVD (4 power iterations, oversampling 10), rows streamed in chunks; numpy and torch agree
  (same Ω); sign convention = largest |component| positive.
- **Factor diagnostics** (`factors.star`, `factors.png`): adjusted multiple correlation of each factor with the
  wedge orientation (second moments of the tilt axis and beam direction in the reference frame, 12 features),
  adjusted correlation ratio with the tomogram, |r| with height and `py3staRefCC`. *Adjusted* matters: with 30
  particles the raw multiple R of 12 random features is ~0.64.
- **Classifiers**: kmeans (k-means++, numpy + torch), Ward (scipy, ≤ 30k particles), GMM (full covariance, EM from
  kmeans, BIC); classes numbered by size; stability = ARI on 80 % subsets + per-class best Jaccard (Hennig).
- **Outputs**: `04_msa/` msa.star (`py3staPC1..n`), scores.npy, pca.npz, eigenvolumes.mrcs (binned), scree.star,
  factors.star, PNGs (scree, eigenvolume projections, factor bars). `05_classify/` per K: `class_<method>_K###.star`,
  `classes_K###.mrcs/.star` (counts, FSC 0.143, Jaccard), `summary.star` (ARI, weakest Jaccard, BIC,
  `py3staWedgeR`), PNGs (scatter, class montages, stability, dendrogram). `classify.use` picks the K for select.

Synthetic (80 particles, ±50° wedge, two states = object ± one blob): wmd PC1 separates the states (accuracy 1.0,
3 seeds), wedge r at chance (adjusted 0.0); plain and plain + wedge: factors 1–5 are wedge factors (r ≈ 0.98) and
none separates the states. Pipeline (32 particles, 4 tomograms): kmeans / ward / gmm K=2 recover the states
exactly (ARI 1.0) and go through select → final.

**MIM019** (6081 particles, RELION poses, `data/mim019_msa_check`, bin 2 = box 36, 30 Å, reference mask,
18 415 voxels; average..classify through the pipeline in 69 s):

| mode | stack + PCA | var % PC1–6 | wedge r PC1–8 | tomo eta PC1 |
|---|---|---|---|---|
| wmd | 21 s + 0.3 s | 1.25, 0.41, 0.39, 0.38, 0.37, 0.34 | 0.08, **0.53–0.55 (PC2–6)**, 0.01, 0.06 | 0.43 |
| plain | 12 s + 0.3 s | 1.31, 1.25, 1.22, 1.14, 1.10, 0.98 | 0.90, 0.72, 0.91, 0.89, 0.59, 0.89, 0, 0 | 0.06 |
| plain + wedge | 12 s + 0.3 s | 1.34, 1.26, 1.26, 1.17, 1.09, 1.00 | 0.90, 0.82, 0.90, 0.90, 0.48, 0.87, 0, 0 | 0.06 |

- wmd removes most of the wedge (plain: six leading factors at r 0.6–0.9 carrying ~1.2 % each); PC2–6 of wmd still
  follow the wedge moderately (r ≈ 0.54, near-equal eigenvalues 0.34–0.41 %, striped eigenvolumes) — likely
  wedge-shaped *noise*: d_i only has support inside W_i, so its noise covariance depends on the orientation.
- **K = 2 (kmeans, factors 1–8) is a quality split**: 3549 / 2532 particles, ARI 1.00 (also gmm 0.99, ward 0.82);
  class 1 reaches **15.0 Å** (FSC 0.143) vs 16.3 Å for all 6081 particles; class 2 34.3 Å. Class 2 contains
  99–100 % of the particles of the three low-CC tomograms (lam7_ts_007, lam4_ts_003, lam5_ts_007: 304 of 313);
  the other tomograms contribute 17–64 %. Half sets: ARI 0.000 for every method and K (control).
- **K ≥ 3 with factors 1–8 sorts by wedge orientation**: class/wedge r 0.35–0.45 (K = 3–8), stability ARI 0.70–0.83,
  weakest Jaccard 0.42–0.60. With factors **1, 7, 8** (wedge factors dropped): class/wedge r ≤ 0.05,
  K = 3 ARI 0.99 (Jaccard 0.98), K = 6 ARI 0.92 (Jaccard 0.88). → classify now reports `py3staWedgeR` per K
  and warns when `classify.factors` include factors msa flagged.
- gmm BIC decreases monotonically up to K = 8 (no minimum): continuous variability, BIC not a K selector here.
- Timing of classify (6081 × 8): kmeans 2.7 s (GPU, 5 K + 10 bootstraps each), ward 24 s, gmm 24 s.

## novaCTF vs Warp tomograms, full cycle (2026-10-17, RTX A5000)

Question (user): are the novaCTF tomograms (`testing_different_recons/xml_tomos/nova_recon_dw`, made from the
Warp XML alignments, dose-weighted) good, and is anything gained? Setup: two projects in `data/novactf_test/`
(warp, nova) from the same STAR — `data/mim019_cycle` cycle-1 input poses with the exclusions cleared (6081
particles), nova with `rlnMicrographName` → `<tomo>.mrc.mrc`; same geometry (1024×1440×500, 6.65 Å, origin 0);
identical cycle.yaml (align bin 2, msa wmd, whitened classify runs, select `use: km_clean/K003`, target each);
full cycle 0 → `next` → cycle-1 average. nova needed its own stack (1010 s for 6081 crops); total ~27 min per arm.

- CTF: `py3sta ctf-check` (300 particles, reference = Warp average, zeros at 34 / 24 Å): Warp +0.51 / flipped
  −0.38, novaCTF +0.45 / −0.37 → both already phase-flipped, `ctf: none` is right (as found before on one
  tomogram). No CTF setting change needed.
- The two reconstructions weight frequencies very differently: relative to the lowest shells the novaCTF
  average has 2.6× Warp's amplitude at 80 Å and 4–7× at 15–50 Å. Real-space CCs (per-tomogram CC/SNR,
  alignment CC, CC with the template) therefore cannot compare the two: they mostly measure this weighting
  (template CC 0.84 Warp vs 0.49 nova). FSC is invariant to it.
- No global offset: nova average onto Warp average = net rotation ~0°, shift < 1 Å, FSC between them 0.99 to
  44 Å, 0.53 at 21 Å.

| | half-map FSC 0.143 | FSC vs template at 30 / 25 / 22 / 19 Å | template FSC 0.5 |
|---|---|---|---|
| Warp, input poses (refined on Warp data) | 16.8 Å | 0.95 / 0.88 / 0.74 / 0.65 | 16.5 Å |
| Warp, after cycle (align: median change 0.00°, 0.5 Å — converged) | 16.9 Å | 0.95 / 0.88 / 0.74 / 0.64 | 16.6 Å |
| novaCTF, input poses | 21.1 Å | 0.93 / 0.88 / 0.68 / 0.61 | 17.7 Å |
| novaCTF, after cycle (align: median change 1.42°, 3.8 Å) | 18.2 Å | 0.94 / **0.91 / 0.76 / 0.72** | 16.5 Å |

- With the Warp-refined poses novaCTF looks worse (21.1 Å); one alignment round on the novaCTF tomograms moves
  the particles (1.4°, 3.8 Å median — the poses were refined on Warp's reconstruction, which has corrections a
  reconstruction from the XML alignment does not) and recovers most of it. After it, the agreement with the
  template is equal (30 Å) or better (25–19 Å) than Warp's; the half-map FSC is still 1.3 Å worse (18.2 vs
  16.9 Å). Half-map FSC measures self-consistency (noise), template FSC measures signal: novaCTF carries more
  high-frequency signal but also more noise, and its poses are not converged yet.
- Classes: same picture as Warp (K003 stable, wedge r ≤ 0.03 whitened, best class 17.7 vs 16.7 Å); select each:
  classes in register (≤ 0.2° net, ≤ 0.3 Å). Same three low-CC tomograms in both after the cycle.
- Cycles 1–3 on each arm (full cycles, novaCTF first, then Warp; `next`; cycle-4 average = result of cycle 3;
  ~9.5 min per cycle):

| arm, cycle average | half-map FSC 0.143 | template FSC 0.5 | FSC vs template 30 / 25 / 22 / 19 / 17 Å | align of that cycle (median) |
|---|---|---|---|---|
| novaCTF 0 | 21.1 Å | 17.7 Å | 0.93 / 0.88 / 0.68 / 0.61 / 0.47 | 1.42°, 3.80 Å |
| novaCTF 1 | 18.2 Å | 16.5 Å | 0.94 / 0.91 / 0.76 / 0.72 / 0.55 | 0.50°, 0.55 Å |
| novaCTF 2 | 18.1 Å | 16.5 Å | 0.94 / 0.91 / 0.76 / 0.71 / 0.55 | 0.00°, 0.27 Å |
| novaCTF 3 | 18.1 Å | 16.5 Å | 0.94 / 0.91 / 0.76 / 0.71 / 0.55 | 0.00°, 0.25 Å |
| novaCTF 4 | 18.0 Å | 16.6 Å | 0.94 / 0.91 / 0.75 / 0.71 / 0.55 | — |
| Warp 0 | 16.8 Å | 16.5 Å | 0.95 / 0.88 / 0.74 / 0.65 / 0.54 | 0.00°, 0.50 Å |
| Warp 1 | 16.9 Å | 16.6 Å | 0.95 / 0.88 / 0.74 / 0.64 / 0.53 | 0.00°, 0.18 Å |
| Warp 2 | 16.9 Å | 16.7 Å | 0.95 / 0.88 / 0.74 / 0.64 / 0.52 | 0.00°, 0.12 Å |
| Warp 3 | 16.9 Å | 16.8 Å | 0.94 / 0.88 / 0.73 / 0.64 / 0.52 | 0.00°, 0.11 Å |
| Warp 4 | 16.9 Å | 16.8 Å | 0.94 / 0.87 / 0.73 / 0.63 / 0.52 | — |

  Both converge by cycle 2 (novaCTF: 2 cycles to move off the Warp-refined poses; Warp: already converged).
  Converged, novaCTF has the worse half-map FSC (18.0 vs 16.9 Å) but the better agreement with the template at
  25–19 Å (0.91 / 0.75 / 0.71 vs 0.87 / 0.73 / 0.63), equal at 30 and 17 Å and at the template FSC 0.5
  (16.6 vs 16.8 Å). Reading: novaCTF carries more signal at 19–25 Å and more noise (its high frequencies are
  4–7× stronger relative to the low ones); half-map FSC rewards Warp's lower noise, the independent template
  rewards novaCTF's signal. Warp drifts very slightly away from the template over cycles (0.65 → 0.63 at 19 Å)
  while self-consistency stays — mild overfitting to its own reference, not seen in novaCTF. Caveat: one
  template (RSTM `template.mrc`); a gold-standard check (independent half-set refinement) or a second
  reference would confirm. No gain in the half-map resolution at this box/pixel size (6.65 Å, Nyquist 13.3 Å).

## Gold standard (`gold_standard: true`) and the MIM019 check (2026-10-17, RTX A5000)

`gold_standard: true` (cycle.yaml): the half sets (rlnRandomSubset 1 / 2) never meet. `02_reference` writes
`references_half{1,2}.mrcs` (the half averages, same mask / band-pass; file references: the same files for both,
warned without a low-pass); `03_align` aligns each half set to its own half reference; `07_select` averages every
class per half set and aligns it to that half's target (reference half, the anchor class's half average for
`each`, or the MRC), transforms per (class, half) in `transforms.star` (`py3staHalf`), `classes_aligned_half{k}`.
Classification stays on all particles (labels do not move particles). The step parameters only carry
`gold_standard` when it is on, so projects from before stay done. Tests: half set 2 (4 of 16 particles) with a
common 9° error — gold standard leaves it at its own pose (< 2°), plain alignment pulls it towards the
half-1-dominated reference (3.4°); select per half restores a perturbed class for target MRC and each (4-particle
half-class averages: ≤ 1.4°); validation (needs average.halves, both half sets); toggling → reference stale.

MIM019 (`novactf_test/gold_{warp,nova}`): from the **original RELION poses** (`mim019_cycle/input.star`,
rlnRandomSubset 2980 / 3101), Warp `ctf: none`, novaCTF `ctf: amplitude`, gold standard, cycles 0–3 + cycle-4
average (~10 min per cycle). CC between the half references 0.995 (Warp) / 0.984 (novaCTF) every cycle.

| run, cycle average | FSC 0.143 | half-map FSC 34/30/25/22/19/17/15 Å | template FSC same shells | align of that cycle |
|---|---|---|---|---|
| Warp c0 (RELION poses) | **16.3 Å** | 0.70 0.73 0.51 0.28 0.20 0.17 0.14 | 0.96 0.98 **0.95 0.87 0.83 0.77 0.67** | 2.0°, 3.3 Å |
| Warp c1 | 16.9 Å | 0.72 0.73 0.48 0.27 0.18 0.16 0.13 | 0.95 0.95 0.88 0.75 0.65 0.54 0.48 | 0.0°, 0.5 Å |
| Warp c4 | 16.9 Å | 0.71 0.73 0.46 0.27 0.17 0.15 0.13 | 0.95 0.95 0.88 0.73 0.64 0.52 0.46 | — |
| novaCTF c0 (RELION poses) | 18.6 Å | 0.73 0.69 0.51 0.20 0.17 0.06 0.04 | 0.98 0.97 0.95 0.82 0.79 0.68 0.53 | 1.7°, 4.2 Å |
| novaCTF c1 | 17.8 Å | 0.76 0.72 0.56 0.28 0.22 0.09 0.06 | 0.98 0.96 0.93 0.78 0.74 0.59 0.46 | 0.5°, 0.7 Å |
| novaCTF c4 | 18.0 Å | 0.75 0.72 0.55 0.26 0.22 0.09 0.07 | 0.97 0.95 0.92 0.76 0.71 0.57 0.44 | — |

- Gold standard changes nothing measurable at this resolution: the converged values equal the shared-reference
  runs (Warp 16.9 Å, template 0.64 vs 0.63 at 19 Å; novaCTF 18.0 Å) — the earlier half-map FSCs were not
  inflated by overfitting.
- novaCTF vs Warp, now with independent halves: the same picture — Warp better half-map FSC (16.9 vs 18.0 Å),
  novaCTF + amplitude better template FSC at 25–17 Å (0.92 / 0.76 / 0.72 / 0.57 vs 0.88 / 0.73 / 0.64 / 0.52).
- **The main finding: py3sta's alignment degrades RELION's poses.** The RELION poses on the Warp tomograms give
  the best map of every run (16.3 Å; template FSC 0.83 at 19 Å, 0.77 at 17 Å); one py3sta alignment cycle drops
  that to 0.65 / 0.54 and it never recovers (gold standard or not, novaCTF or Warp). This is the 0.5 Å loss of
  Phase 2 part 3 / Phase 5, but the template FSC shows it is large at 15–25 Å. Likely cause: alignment at bin 2
  (13.3 Å pixel) with a 30 Å low-pass cannot see the 15–25 Å signal that defined RELION's poses, and its
  angular / shift steps (0.5°, sub-voxel at bin 2) add pose noise there. Next: align at higher resolution
  (bin 1, low-pass ~18–20 Å, finer last level), and/or keep a particle's pose unless the CC improves.

## `ctf: amplitude` on novaCTF and Warp tomograms (2026-10-17, RTX A5000)

`ctf.mode` acts on averages only (cycle average, class averages, the reference made from them); alignment and
MSA use the wedge only (CTF there is Phase 7). So `amplitude` (|CTF| weighting in the Wiener average) can act
on the poses only through the reference.

1. Same poses (each arm's converged cycle-4 input), `py3sta average` from the tomograms, `none` vs `amplitude`
   with the per-particle height sign +1 / −1 / 0: amplitude raises the half-map FSC by 0.01–0.04 around the
   first CTF zeros (34–25 Å) and the template FSC by ~0.03 at 34 Å (Warp also +0.02–0.03 at 25–19 Å); FSC 0.143
   18.0 → 17.8 Å (novaCTF), 16.9 → 16.8 Å (Warp). The height sign makes no difference at this resolution (all
   three within 0.01) — kept at the default +1.
2. Two more full cycles from the converged poses: `amplitude` in copies (`novactf_test/{nova,warp}_amp`, own
   stack copies) vs `none` in the original projects (cycles 4–5, averages 4–6):

| run, cycle average | FSC 0.143 | half-map FSC 34/30/25/22/19/17/15 Å | template FSC same shells | align of that cycle |
|---|---|---|---|---|
| novaCTF none c5 / c6 | 18.1 / 18.1 Å | 0.75 0.70 0.53 0.25 0.20 0.08 0.06 | 0.94 0.94 0.91 0.75 0.71 0.55 0.42 | 0.0°, 0.2 Å |
| novaCTF amplitude c1 / c2 | 18.0 / 18.0 Å | 0.78 0.73 0.55 0.26 0.21 0.08 0.06 | 0.97 0.95 0.92 0.75 0.71 0.56 0.43 | 0.5°, 0.3–0.6 Å |
| Warp none c5 / c6 | 16.9 / 16.9 Å | 0.73 0.73 0.47 0.27 0.17 0.16 0.13 | 0.94 0.94 0.87 0.73 0.63 0.52 0.45 | 0.0°, 0.1 Å |
| Warp amplitude c1 / c2 | 17.9 / 18.0 Å | 0.77 0.74 0.46 0.26 0.15 0.14 0.11 | 0.96 0.95 0.87 0.70 0.63 0.51 0.42 | 0.7° / 1.2 Å, then 0.0°, 0.4 Å |

   novaCTF: a small, consistent gain (low shells +0.02–0.03 half-map, +0.03 template; high shells equal or
   +0.01) — use `amplitude` for novaCTF. Warp: the same-pose average is unchanged, but the amplitude-weighted
   reference moves the particles (0.7°, 1.2 Å, alignment CC 0.36 → 0.34) and the next averages lose 1 Å
   (16.9 → 18.0) and template FSC at 22–15 Å — keep `none` for Warp. Likely Warp's reconstruction already
   carries CTF amplitude weighting, so weighting again over-suppresses the shells near the zeros. Note
   novaCTF+amplitude keeps moving particles by a median 0.5° per cycle (not 0.0° as with none): the poses
   follow the slightly different reference without a measurable gain or loss.
   (A first attempt failed between cycles because the pipeline code changed under the running jobs — the
   select parameters gained a key and the finished cycle looked stale; re-run from `next`.)

## Wedge whitening of the factor scores (2026-10-16)

User decision (the parked Phase 3 question): option 2, whitening; flagged factors and wedge-following classes
are only warned about (nothing excluded automatically); WedgeR is reported so the dashboard can later let the
user avoid such factors / classifications. Default on (user).

- `msa.whiten_wedge(X, mats)`: per factor, least squares of the scores on the 12 wedge features (rank 10:
  the traces of a aᵀ and z zᵀ are 1) + intercept; the fitted wedge part is subtracted, mean kept. Only factors
  whose fit beats chance (adjusted R² > 0) are changed: a 12-feature fit explains ~p/n of any column by chance —
  on the 32-particle test project it removed the state signal before this guard. Returns the variance fraction
  removed per factor. Linear only: non-linear wedge dependence stays, so the class WedgeR check stays.
- `classify` runs and `embed`: `wedge: whiten | keep` (default whiten), applied before `standardize` / UMAP;
  part of the run key (changing it recomputes only that run). Each whitened run writes `wedge.star`
  (py3staWedgeRemoved per factor); `summary.star` / `compare.star` get `py3staWedge` and `py3staWedgeRemoved`
  (variance-weighted fraction of the run's factors); the manifest info has it per classification. Warnings: the
  msa-flagged-factors warning says "whitened … check the classes' WedgeR" (or suggests whiten / dropping them
  with keep); class WedgeR > 0.3 still warns; fewer than 240 particles (20 per feature) warns that whitening
  is noisy.
- Tests: synthetic two states under a 6× stronger wedge component (600 random poses): raw kmeans ARI < 0.2 with
  class WedgeR > 0.5; whitened ARI > 0.95, WedgeR < 0.1; pure-noise column untouched (< 5 %); means kept; no
  linear wedge correlation left. Pipeline: default whiten, `wedge.star`, compare columns, `keep` → stale,
  old warning text, validation.

MIM019 (`data/mim019_cycle` cycle 0 after alignment, 6081 particles, kmeans, 10 bootstrap subsets; 154 s for
6 runs):

| run | wedge | removed | K003 WedgeR / weakest Jaccard / best class | K004 WedgeR / weakest Jaccard / best class |
|---|---|---|---|---|
| factors 1–8 | keep | — | **0.34** / 0.75 / 18.8 Å | **0.41** / 0.58 / 21.0 Å |
| factors 1–8 | whiten | 16 % (factors 2–6: 29–32 %; 1: 1 %, 7–8: 0 %) | 0.03 / 0.97 / 16.7 Å | 0.18 / 0.79 / 16.7 Å |
| factors 1, 7, 8 | keep | — | 0.05 / 0.97 / 16.6 Å | 0.05 / 0.91 / 19.1 Å |
| factors 1, 7, 8 | whiten | 0.7 % | 0.02 / 0.98 / 16.7 Å | 0.02 / 0.80 / 19.1 Å |
| UMAP, factors 1–8 | whiten | 16 % | 0.04 / 0.99 / 16.5 Å | 0.03 / 0.93 / 16.6 Å |

Agreement (ARI) at K003: whitened 1–8 vs the hand-picked clean factors 0.97 (whitened) / 0.93 (kept); kept 1–8
vs clean 0.56. Whitening makes all eight factors usable without choosing them by hand: the wedge-driven K003 /
K004 splits disappear, stability and the best class improve, and the clean factors barely change. K004 on
whitened 1–8 keeps WedgeR 0.18 (non-linear rest; below the 0.3 warning). The 29–32 % removed from factors 2–6 matches
their msa wedge r ≈ 0.54 (R² ≈ 0.29); the remaining 70 % of those factors no longer follows the wedge linearly.

## Phase 6 v1 — read-only dashboard (2026-10-16)

User decisions: Dash (default of PLAN §4); shared workstation → access token, one job queue per project (v2).
`pip install 'py3sta[dashboard]'` (dash 2.18.2 + plotly; installed into `tomo_toolshed`).

- `py3sta dashboard PROJECT [--port 8050] [--host 127.0.0.1] [--token T | --no-token]`: listens on 127.0.0.1 by
  default (SSH tunnel from a laptop; the command prints the `ssh -L` line); a random token is printed in the
  URL; requests without it get 403, the first `?token=` sets an httponly SameSite cookie. Refuses a busy port.
- `dashboard/data.py` reads everything through the pipeline API (project_status, manifests, STAR/MRC); no Dash
  imports, unit-tested. `dashboard/app.py` renders; render functions are plain and tested directly, callbacks
  also through Dash's HTTP endpoint.
- Pages: **Cycles** (one row per cycle: particles, excluded, ctf, FSC 0.143 first/last crossing, projections
  thumbnail of the average as an inline PNG — a 20-line zlib encoder, no imaging library — and a coloured badge
  per step with the stale reasons / time / device / warnings as tooltip; resolution-per-cycle plot; refreshes
  every 10 s so CLI runs show up), **Maps** (any map of a cycle: XY/XZ/YZ slices with position sliders or
  projections, low-pass in Å, reference-mask contour, volume slider + gallery for stacks; volumes are read
  server-side and cached by mtime, only 2D views are sent), **Plots** (half-map FSC of every cycle, scree,
  sortable `compare.star` table, every step PNG of the cycle — served by `/files/...png`, PNG only and inside
  the project), **Logs** (log.txt, warnings, error + traceback of any step).
- `run_step` now writes `log.txt` in each step folder (time-stamped, flushed per line so it can be followed
  while running; on failure the error and traceback are appended). Steps run before this have no log.txt.
- ChimeraX (`chimerax.py`, button in Maps, `py3sta chimerax MAP...`): REST remote control
  (`remotecontrol rest start port 60000` in ChimeraX). Checked against ChimeraX 1.10.1 headless: `json true`
  replies to `open` with "'Volume' object has no attribute 'string'" although the map opened — treated as
  success; stacks (`.mrcs`) open as one tall map, so they are split into one model per volume (`volume copy
  subregion`, origins side by side along x, renamed `file [i]`; `tile` fails headless). MIM019: average,
  21 eigenvolumes, 3 aligned classes opened (25 models). ChimeraX must see the files (same machine or
  shared filesystem; reverse tunnel for the port).
- MIM019 (`data/mim019_cycle`): overview 0.7 s (6081 particles, two cycles), slices 0.1 s, 21-volume gallery
  0.1 s, plots page 0.14 s. Not yet looked at in a real browser (no browser here) — callbacks checked over HTTP.
- Tests: PNG round trip, views, log.txt (success + failure), overview states / stale reason / FSC / thumbnail,
  token 403/200/cookie, busy port, maps (slice through a known voxel in all three views, stack index, mask
  contours, gallery), PNG route (no MRC, no `..`), callbacks over HTTP, ChimeraX against a fake REST server
  with the captured replies (incl. the encoding error), not running, dashboard button.

## Phase 5 — `select.target: each`, full MIM019 cycle (2026-10-16, RTX A5000)

**`target: each`** (user decision: anchor = best-SNR class, not a running average). The anchor is the align
class with the best class resolution (`py3staClassRes`, half-map FSC 0.143 from classify), ties → the larger
class; class 0 (HDBSCAN noise / not in a lasso group) only if it is the only align class. `select.anchor: N`
overrides (validated: needs `target: each`, must be an align class). The anchor's members keep their poses;
the other align classes are aligned to the anchor's average (same `align_maps` search, reference mask).
`transforms.star` has `py3staClassAnchor`; the manifest info has `anchor`.
Tests: the chiral 2-class project with class 2 perturbed by a known (R, t): with the automatic anchor and with
`anchor: 2` all particles end up within 1° of one common transform of the true poses, the anchor members'
poses are unchanged; `pick_anchor` (noise skipped, excluded classes skipped, ties, no resolution → largest);
validation; `anchor` change → select stale. 155 tests.

**Full cycle on MIM019** (`data/mim019_cycle`, 6081 particles, shared stack). Cycle 0: average (7 s) →
global reference (sphere 191.5 Å) → align on (default levels `[[12,4,12,4],[3,1,3,1],[1,.5,1,.5]]`, bin 2,
low-pass 30 Å; 342 s; median change 2.06°, 3.3 Å; 614 particles at the search edge) → msa wmd (25 s; factors
2–6 follow the wedge, as before) → classify `km_clean` (kmeans, factors 1/7/8) + `umap_km`, K 2–4 (68 s; all
wedge r ≤ 0.08, stable: Jaccard ≥ 0.91) → select `use: km_clean/K003`, `target: each` (39 s) → final → `next`
→ cycle 1 average.

K003 classes: 1 = 2390 particles, 35.2 Å (holds the three low-CC tomograms); 2 = 2211, 16.6 Å (anchor);
3 = 1480, 22.3 Å. Class transforms onto the anchor: class 1 rot/tilt/psi 51.8/0/−51.8 (= 0° net), class 3
0.1°, shifts 0.0–0.1 Å, CC 0.95–0.96 → the classes were already in register (every particle was aligned to
the same reference in 03_align), so `each` changes nothing here, as expected for one-reference cycles.
After alignment the K=2 split is 3556/2525 (not the 5777/304 quality split of the unaligned msa check).

| | particles | FSC 0.143 (first crossing) | CC vs `template.mrc` |
|---|---|---|---|
| cycle 0 average (RELION input poses) | 6002 | 16.3 Å | 0.845 |
| run A: cycle 1, all classes aligned | 6001 | 16.8 Å | 0.840 |
| run B: cycle 1, class 1 excluded | 3658 | 16.3 Å | 0.818 |

(CC: `py3sta align-maps AVERAGE template.mrc`, sphere 191.5 Å soft 47.9 Å, no contrast flip — the averages
are already protein-white like the template.) Acceptance "next cycle ≥ previous": run A misses by 0.5 Å, the
same loss Phase 2 part 3 measured for aligning from RELION's poses (16.3 → 16.9 Å in the control run), so it
comes from 03_align, not select. Run B keeps the resolution with 40 % fewer particles (the excluded class is
the low-quality one: no low-CC tomograms remain); its lower CC vs the template is expected from the smaller
average (more noise) and is not a resolution loss.

Open: how to make cycle 1 ≥ cycle 0 when the input poses are already refined — e.g. finer last level / no
first-level jump, align only to the best class, or a per-particle "keep the old pose unless the CC improves"
rule (`align.min_gain` exists, default 0).

## Phase 4 — `06_embed` + `py3sta explore` (lasso groups) (2026-10-16)

- `06_embed` (`embed:` umap | tsne | pca on `embed.factors`, default 1–8; upstream msa only, skipped when msa is
  disabled): `embedding.star` (`py3staEmbed1/2`), `embedding.png` (tomogram, PC, alignment CC).
- `py3sta explore PROJECT CYCLE` → `06_embed/explore.html`: self-contained (plotly.js inlined from the `plotly`
  package — user's choice: offline over CDN; 6.5 MB for MIM019, built in 0.6 s). Axes: the 06_embed map, any two
  PCs, every classify run's UMAP; colours: every classification, tomogram, half set, PCs, alignment CC,
  |beam x/y/z| in the reference frame (wedge), height. Lasso/box → named groups (last selection wins, moves
  reported), rename, delete, undo, per-group main tomograms; export / reload a groups file.
- Groups enter as a classify run (user's choice) `{method: groups, file: groups/NAME.json}`: group k = class k,
  class 0 = in no group, class names in `classes_all.star`; cached by the file's sha1 (the file is also a tracked
  input: editing it makes classify stale).
- Tests: embed modes (two-state map: k-means on the 2D map = true states for umap, tsne and pca); the page's data
  round-trips; `tests/explore_harness.js` runs the page script in node on a stub DOM (two overlapping lassos,
  undo, export) and the exported file loads back; acceptance: a polygon around one cluster of the map → groups
  file → classify → select (exclude class 0) → final reproduces the true states. Headless Firefox (snap) could
  not render the page here: the plot itself is not checked in a browser yet.
- MIM019 (6081 particles): UMAP of factors 1–8 in 28 s; two map clusters (3437 / 2644). A polygon around cluster 1
  (its hull) picks 3417 particles, none of cluster 2; as a classification it agrees with every K=2 classification
  (ARI 0.88–0.90; halves 0.00), group class 15.1 Å. Re-running classify with nothing changed: 8 runs cached, 2 s.
  (Adding fields to `ClassifyRun` changes run keys → one full recompute; irrelevant fields are dropped from the key.)

## Classification runs: named runs, UMAP / HDBSCAN, comparison (2026-10-16)

`classify:` flat settings = the defaults; `classify.runs` = {name: {settings}} (or a list; no name → automatic,
e.g. `kmeans_umap_f1_7-8`); `classify.use` = `run/K004` | `run` | K. Each run in `05_classify/<run>/` with a
`run.json` key (run settings + msa run id + pose STAR sha1 + averaging settings): only new or changed runs are
recomputed (delete a run folder to force). `compare.star/.png` (stability, class/wedge r, class/tomogram V, best
class FSC 0.143, BIC, noise) and `agreement.csv/.png` (ARI between all classifications). `space: umap`
(umap-learn, seeded) clusters a UMAP embedding of the factors; `method: hdbscan` (scikit-learn) finds K and a
noise class 0. Bootstrap on UMAP runs re-clusters subsets of the fixed embedding.

MIM019 (`data/mim019_msa_check`, RELION poses, wmd factors, 10 bootstraps; 6 runs / 17 classifications with
class averages: 155 s; adding a 7th run: 57 s, 5 runs cached):

| run (factors) | K=2 | K=3 | K=4 | K=6 |
|---|---|---|---|---|
| km_all (1–8) | J 1.00, w 0.07, 15.0 Å | J 0.60, **w 0.35**, 18.7 | J 0.56, **w 0.42**, 19.1 | J 0.42, **w 0.45**, 21.3 |
| km_clean (1,7,8) | 1.00, 0.07, 15.0 | 0.98, 0.05, 16.3 | 0.74, 0.05, 16.4 | 0.88, 0.05, 18.7 |
| gmm_clean (1,7,8) | 1.00, 0.07, 15.1 | 0.92, 0.05, 16.3 | 0.87, 0.04, 16.5 | — |
| umap_km (1,7,8) | 1.00, 0.08, 15.0 | 0.99, 0.05, 16.3 | 0.99, 0.05, 16.4 | 0.91, 0.04, 18.7 |
| umap_km_all (1–8) | 1.00, 0.07, 15.1 | 0.92, 0.07, **15.0** | 0.93, 0.28, 16.9 | — |

(J = weakest class Jaccard, w = class/wedge r, Å = best class FSC 0.143.) HDBSCAN on UMAP (1,7,8,
min_cluster_size 100): 2 classes + 79 noise, J 0.99, 15.1 Å, ARI 0.88–0.96 with every K=2 classification.
Control `halves` (column rlnRandomSubset): ARI 0.00 with everything, each half 17.8 Å. All K=2 classifications
agree (ARI 0.88–1.00): the good / bad split is method-independent. UMAP on all 8 factors is far less wedge-driven
than k-means on them (K=3 w 0.07 vs 0.35) and gives the best K=3 class (15.0 Å, 3439 particles).

Fixes found on the way: a class made of one half set (column rlnRandomSubset) had an empty FSC half (reported
the box size, 478.8 Å) → such classes use alternate rows; the STAR reader did not parse `nan` (pandas
`to_numeric` refuses it) → columns with missing values came back as text.

## `py3sta variance` (2026-10-16)

On-demand voxel-wise variance of a cycle's aligned particles (not a pipeline step; user's decision: no automatic
mask making). Each particle normalised inside `--norm-mask` (a homogeneous part, e.g. a membrane); variance of
the wmd differences (default) or plain; optional `--mask` restricts the report (may equal the normalisation
mask); stats vs the region outside both masks. Synthetic: wmd peaks at the variable blob (> 4× median);
one state: wmd object/solvent variance 1.15 vs plain 5.2 (plain + own wedge 6.3) — the plain map shows the wedge.
MIM019 (normalised in the reference mask, bin 2, 30 Å, 6081 particles, ~20 s): variance lowest in the particle core
(~0.75 normalised units), ~0.95 at the periphery / neighbours; plain adds a bright ring at the particle boundary.
With a whole-particle normalisation mask the map is mostly noise — the tool is meant for targeted masks.

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
- **Options tried (2026-10-16)**, all started from the RELION poses (bin 2, 30 Å, shift 2 voxels unless noted):

| run | rotation change (median) | centre change (median) | FSC 0.5 | FSC 0.143 |
|---|---|---|---|---|
| RELION poses, not aligned | — | — | 24.9 Å | 16.3 Å |
| baseline: full cones | 2.06° | 3.35 Å | 25.8 | 16.9 |
| 1a local cones only | 2.00° | 3.11 Å | 25.4 | 16.7 |
| 1b min_gain 0.01, local cones (whole pose kept) | 0° | 0 Å | 25.1 | 16.3 |
| 1c min_gain 0.02, full cones (whole pose kept) | 0° | 0 Å | 25.1 | 16.3 |
| 2 bin 1, reference + search at 20 Å, local | 2.00° | 2.97 Å | 24.6 | 16.8 |
| 3 half-set references (half 1 → half1.mrc …) | 2.06° | 3.32 Å | 25.7 | 16.9 |
| shifts only | 0° | 2.42 Å | 25.6 | 16.4 |
| **rotations only, local cones** | 1.83° | 0 Å | **24.3** | 16.6 |
| perturbed start, full cones + min_gain 0.02 | 2.05° | 2.34 Å | 25.6 | 16.8 |

  - **The shifts cause the FSC loss, not the rotations**: rotation-only local refinement beats RELION at
    FSC 0.5 (24.3 vs 24.9 Å). Shift estimates are unbiased (mean ≈ 0) but noisy and anisotropic: rms 1.9 /
    2.7 / 3.9 Å along tomogram x / y / z (missing wedge elongates the CC peak along z); low-CC particles
    shift most (median 5–6 Å in the lowest CC quarter). RELION refines against the tilt images (user's point),
    which pins the centres better than subvolumes from reconstructed tomograms can.
  - Half-set references: no effect at this resolution. Bin 1 with matched filters: small gain at 0.5 only,
    ~9x slower. min_gain (whole pose kept) preserves good poses but does not help the perturbed set.
  - Acted on: `align.max_shift` per tomogram axis (`[x, y, z]` Å, e.g. a small z limit; `0` = rotations
    only) and `align.min_gain` (opt-in restraint, default 0). min_gain first kept only the rotation and still
    re-estimated the shift — fixed to keep the whole pose.

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

- (2026-10-16) `classify.use` (in `classify:`) picks the classification passed to select (`run/K004`, `run`, or K);
  it is select's parameter (changing it does not re-run classify). Default method kmeans, factors 1–8,
  K = [2, 3, 4, 6, 8], 20 bootstraps.
- (2026-10-16) Several named classification runs per cycle (user-named; automatic names when missing), class
  averages for every run × K by default (`averages: false` per run), UMAP space and HDBSCAN in classify.
- (2026-10-16) Diagnostic PNGs now (matplotlib optional); interactive views go to the dashboard.
- (2026-10-16) Plain MSA "with the wedge filter" = each particle filtered with its own wedge, the mean is not.
- (2026-10-16) `py3sta variance` is a CLI command on demand and does not make masks.

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

- (2026-10-16) wmd factors 2–6 on MIM019 still follow the wedge (r ≈ 0.54): exclude flagged factors
  automatically (`classify.factors: auto`), or whiten the differences per wedge (e.g. divide by the expected noise
  power of W_i)? For now: a warning and `py3staWedgeR` per K.

- Dedicated py3sta conda env (user will create it); then update CLAUDE.md.
