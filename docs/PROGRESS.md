# py3sta — Progress

Update at the end of every work session: move items, add findings with numbers, note open questions.

## Current focus

**Phase 1 — pipeline skeleton** (see PLAN.md §4). Nothing started yet.

Next concrete tasks:
1. `project.yaml` / `cycle.yaml` schema + loader with validation; `py3sta init`.
2. Cycle folder + `manifest.json` + `py3sta status` (done / stale / missing).
3. Reference step (global, file) with masks + band-pass in Å.
4. Stack step: pre-rotated `stack.mrcs` (float16) + `stack.star`, CPU multiprocess + torch.
5. Pose composition function + synthetic tests; `_final.star` writer; `py3sta next`.
6. Dummy classification from existing `rlnClassNumber` to run select/final end-to-end before the real engines exist.
7. `py3staExcluded` column: honoured by every step from the start (average already needs a filter), `py3sta flag`.

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

- Package/CLI name `py3sta` (GitHub `hamid13r/py3sta`); renaming later is cheap (SAGA was the favourite alternative).
- RELION 4 particle format; Warp export via minimal STAR (`star2warp`).
- Tomogram path in `rlnMicrographName`, Warp XML in `rlnWarpXMLfile`; no `--tomo-dir`.
- User wants PCA, k-means and UMAP, with lasso drawing on the UMAP plot to define particle groups (Phase 3–4).
- Excluded particles are **kept** in the STAR file with `py3staExcluded = 1`, carried over to all later
  cycles and skipped by every step, until the user flips the flag back (`py3sta flag --include`). (PLAN.md §2)
- GPU backend confirmed working by the user with `py3sta average --backend torch` (CUDA 12.2).

## Open questions

- Numerical parity numbers from `scripts/check_torch.py` on the GPU not yet recorded (backend runs; record diffs + timing).
