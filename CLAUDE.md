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
  cli.py             click CLI: init pos2star trf2star star2trf star2warp average add-xml ctf-check wedge-check
  io/                star.py (dependency-free STAR), mrc.py (numpy MRC), trf.py (i3), dipole.py (.pos picks),
                     particles.py (ParticleSet = RELION-4 particle table + tomo_key naming)
  pipeline/          config.py (project.yaml / cycle.yaml schema + validation), project.py (init, load, cycles)
  utils/             stand-alone utilities (each also `python -m py3sta.utils.<name>`), e.g. add_warp_xml
tests/               pytest; scipy/torch tests auto-skip when not installed
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
- **Parallel CPU:** `average_particles(jobs=N)` uses processes; avoid BLAS calls in workers (oversubscription
  made -j4 slower than -j1 once — use element-wise math).
- **Units:** Å for physical sizes in user-facing options; voxels internally; angles in degrees.

## Working rules

- numpy implementation is the reference; any torch code needs a parity test (`importorskip("torch")`).
- Every new physical convention needs (1) a synthetic test and (2) a real-data check on MIM019, recorded in PROGRESS.md.
- Keep the CLI thin: logic lives in modules, CLI only parses, resolves files and reports. The planned
  dashboard (PLAN.md §4) must call the same module API, so never put processing logic in CLI or GUI code.
- New per-particle columns: use `rln*` names only when they mean exactly the RELION label; otherwise `py3sta*`
  (exception: `rlnWarpXMLfile`, chosen by the user).
- Git: small commits with clear messages; never commit `data/`, MRC files or caches.

## Test data (local, not in git)

`/data1/users/hamid/robpicker_MIM019/py3sta_test/` —  ~6K particles (relion export, 6.65 Å, box 72, has rlnRandomSubset).
Tomograms + Warp XML + `average.mrc` 
Results of earlier checks: `data/average_test/`.
