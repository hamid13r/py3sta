# CLAUDE.md — py3sta

GPU-accelerated, open-source re-implementation of **i3** subtomogram averaging ("alignment through
classification"), built on **RELION 4 STAR** files and compatible with **Warp/WarpTools**.
Author: Hamidreza Rahmani (GitHub `hamid13r`, repo `py3sta`). The i3 workflow being replaced is documented
in `../i3_protocol/` (cycle steps, parameters, pitfalls).

Read these before working:
- `docs/PLAN.md` — roadmap, architecture, phase-by-phase tasks with acceptance checks
- `docs/PROGRESS.md` — what is done, what is next, decisions and empirical findings (update it at the end of every session)

## Layout

```
src/py3sta/
  geometry.py        Euler <-> matrix (RELION ZYZ), frame_from_normal          <- single source of rotation conventions
  wedge.py           TiltInfo (Warp XML / .tlt), missing wedge, dose, CTF (CTFParams, particle_defocus, ctf_transfer)
  average.py         extract (trilinear), average_particles (multiprocess, wedge/dose/CTF Wiener), fsc, path resolution
  backend_torch.py   PyTorch/CUDA mirror of extract/wedge/average (must stay numerically equal to numpy)
  poses.py           pose composition: delta (R, t) in the reference frame -> M' = R M, c' = c + M'.T t
  mra.py             particle alignment: cone/spin levels, constrained CC (wedge on both maps), best reference
  msa.py             MSA stack at the current poses (wmd = wedge-masked differences | plain ± own wedge),
                     randomized PCA (numpy + torch), wedge / tomogram factor diagnostics
  variance.py        voxel-wise variance of the aligned particles (wmd / plain), normalised in a chosen mask
  cluster.py         kmeans (numpy + torch), Ward, GMM (BIC), HDBSCAN, UMAP embedding, classes by size, ARI,
                     bootstrap stability
  plots.py           diagnostic PNGs of the steps (matplotlib optional); interactive views go to the dashboard
  align.py           map-to-map alignment: resample, SO(3) grid, masked CC over shifts (numpy + torch), polish, mirror
  reference.py       masks (sphere/ellipsoid/box/file, soft edges in Å), band-pass, fit_box, prepare_reference
  stack.py           project particle stack: unrotated float16 crops, refresh_stack (new / tomogram changed / drift)
  cli.py             click CLI: init run status next flag variance explore stack align-maps pos2star trf2star star2trf star2warp average add-xml ctf-check wedge-check
  io/                star.py (dependency-free STAR), mrc.py (numpy MRC), trf.py (i3), dipole.py (.pos picks),
                     particles.py (ParticleSet = RELION-4 particle table + tomo_key naming)
  pipeline/          config.py (yaml schema), project.py (init, load, cycle input checks, next_cycle), flags.py (flag), variance.py (py3sta variance),
                     manifest.py (run_step, change tracking, states), steps.py (step order, pose sources,
                     01_average, 02_reference, 03_align, 04_msa, 07_select, 08_final), classify.py (05_classify:
                     named runs, per-run cache, comparison; method groups = lasso groups), embed.py (06_embed),
                     explore.py (py3sta explore: offline HTML, lasso groups file), runner.py (run_cycle, status)
  utils/             stand-alone utilities (each also `python -m py3sta.utils.<name>`), e.g. add_warp_xml
tests/               pytest; scipy/torch/umap tests auto-skip when not installed; explore_harness.js runs the
                     explore page's script on a stub DOM in node (simulated lasso + export)
scripts/check_torch.py   GPU vs numpy parity + timing on real data
data/                local test data and results — git-ignored, never commit (tomograms are GBs)
```

## Commands

For now everything runs in the micromamba env `tomo_toolshed` (`conda activate tomo_toolshed`).
It has `torch==2.5.1+cu121` (cu13x builds do not work with the CUDA 12.2 driver) and `pyyaml`.
A dedicated py3sta env will replace it later.

```bash
pip install -e ".[test]"            # + torch: pip install torch --index-url https://download.pytorch.org/whl/cu121  (CUDA 12.2 driver)
pytest tests                        # must pass before every commit
py3sta --help
python scripts/check_torch.py STAR TOMO --tilt XML --gpu 0
```

## Conventions (do not change without updating tests + PROGRESS.md)

- **Particle table** = RELION 4: `rlnTomoName`, `rlnCoordinateX/Y/Z` in pixels at `ts_apix`
  (0 = centre of first voxel, Warp convention), `rlnOriginX/Y/ZAngst` (true centre = coord − origin/apix),
  `rlnAngleRot/Tilt/Psi`. Pixel size from `rlnTomoTiltSeriesPixelSize` → `rlnMicrographPixelSize` → `rlnImagePixelSize`.
- **Rotation:** matrix `M` maps tomogram → reference; `M = Rz(rot)·Ry(tilt)·Rz(psi) = A_relion.T`;
  extraction samples `tomo(c + M.T @ r)`. Verified against RELION source, scipy, a synthetic asymmetric object,
  and the MIM019 average vs `template.mrc` (mirror and transposed conventions score clearly worse).
- **Per-tomogram files come from the STAR file:** tomogram = `rlnMicrographName`, Warp XML = `rlnWarpXMLfile`
  (paths absolute, or relative to cwd, or to the STAR file's directory). `tomo_key()` maps
  `TS_01.mrc_6.65Apx.mrc`, `TS_01.mrc.tomostar`, `TS_01.mrc.xml` → `TS_01`.
- **Missing wedge:** tilt axis = tomogram Y; sampled if elevation atan(kz/kx) ∈ sign·[tilts]. Warp XML with
  `AreAnglesInverted=True` → sign −1 (measured; `py3sta wedge-check`).
- **CTF:** per-tilt defocus from Warp `GridCTF` + particle height; nearest-tilt 3D CTF; transfer is positive
  at low frequency. Modes: `flip`/`wiener` for uncorrected WBP (IMOD/etomo), `none`/`amplitude` for
  already phase-flipped tomograms (Warp, novaCTF). Decide with `py3sta ctf-check` — **half-map FSC cannot**.
- **Averaging:** Fourier Wiener `Σ w·F / max(Σ w², eps·n)`; half sets from `rlnRandomSubset`, else alternate rows.
  Subvolumes are zero outside the extraction sphere (radius box//2 − 1) and normalised per particle inside
  `particle_radius`. The stack (`stack/`) holds unrotated crops; poses are applied on use (one interpolation).
- **Contrast:** `invert_contrast` (project.yaml) flips densities on read → protein white on black everywhere
  downstream; the stack keeps raw tomogram values. External reference files have their own `reference.invert`.
- **Parallel CPU:** `average_particles(jobs=N)` uses processes; avoid BLAS calls in workers (oversubscription
  made -j4 slower than -j1 once — use element-wise math).
- **Units:** Å for physical sizes in user-facing options; voxels internally; angles in degrees.

## Working rules

- numpy implementation is the reference; any torch code needs a parity test (`importorskip("torch")`).
- Every new physical convention needs (1) a synthetic test and (2) a real-data check on MIM019, recorded in PROGRESS.md.
- Keep the CLI thin: logic lives in modules, CLI only parses, resolves files and reports. The planned
  dashboard (PLAN.md §4) must call the same module API, so never put processing logic in CLI or GUI code.
- New per-particle columns: use `rln*` names only when they mean exactly the RELION label; otherwise `py3sta*`
  (exceptions chosen by the user: `rlnWarpXMLfile`, `rlnBestReference`).
- Git: small commits with clear messages; never commit `data/`, MRC files or caches.

## Test data (local, not in git)

`/data1/users/hamid/robpicker_MIM019/py3sta_test/` —  ~6K particles (relion export, 6.65 Å, box 72, has rlnRandomSubset).
Tomograms + Warp XML + `average.mrc` 
Results of earlier checks: `data/average_test/`.
