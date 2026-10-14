# py3sta

GPU-accelerated, i3-style subtomogram averaging (alignment through classification),
built on RELION 4 STAR files and compatible with WarpTools.

Status: I/O, extraction, wedge-weighted averaging (numpy + PyTorch/CUDA). Alignment/MSA/classification next.

## Install

```bash
pip install -e ".[test]"
pytest tests
```

## Converters (replace `pos2i3.py` and `trf2star.py`)

```bash
# dipole picks -> RELION 4 particles + WarpTools STAR + an i3 project (to keep using i3 meanwhile)
py3sta pos2star pos/ --pick-apix 13.3 --ts-apix 1.33 --random --seed 0 \
      -o particles.star --warp particles_warp.star \
      --i3-dir . --tomo-file-fmt "{tomo}.mrc_13.30Apx.mrc" --tlt-template template.tlt

# i3 output (trf or i3datalist -cls) -> RELION 4 (+ Warp)
py3sta trf2star cycle04-004.trf --tomo-apix 13.3 --ts-apix 1.33 -o c04.star --warp c04_warp.star

# RELION 4 -> i3 project, or -> WarpTools
py3sta star2trf particles.star --tomo-apix 13.3 -o i3_project/
py3sta star2warp particles.star -o particles_warp.star
# then: WarpTools ts_export_particles --input_star particles_warp.star --coords_angpix <printed value> ...
```

## Averaging

Tomograms are read from `rlnMicrographName` (absolute, or relative to the current directory or to the
STAR file). Listing tomogram files after the STAR file overrides this.

Tilt information (wedge, dose) is read per particle from `rlnWarpXMLfile`. Add it with the utility:

```bash
py3sta add-xml particles.star warp_tiltseries/ -o particles_xml.star     # dirs, files or globs
python -m py3sta.utils.add_warp_xml particles.star "warp_tiltseries/*.xml" # same, standalone
```

```bash
py3sta wedge-check TOMO.mrc TOMO.mrc.xml          # which tilt-angle sign matches the data (Warp: -1)
py3sta average particles_xml.star --box 72 --dose-weight --backend torch --gpu 1 -o avg.mrc
python scripts/check_torch.py particles.star TOMO.mrc --tilt TOMO.mrc.xml   # GPU vs numpy check + timing
```

### CTF

`--ctf` uses the per-tilt defocus in the Warp XML (`rlnWarpXMLfile`), shifted by each particle's height in
every tilt, and applies the CTF of each Fourier voxel's nearest tilt (3D CTF model):

| tomogram | `--ctf` |
|---|---|
| not CTF-corrected (IMOD/etomo WBP) | `flip` or `wiener` |
| already phase-flipped (Warp `ts_reconstruct`, novaCTF) | `none` (default) or `amplitude` |

`py3sta ctf-check particles_xml.star reference.mrc --box 72` tells which case applies by comparing averages
with a CTF-corrected reference between the first two CTF zeros. Half-map FSC cannot tell (both halves share
the same CTF error). On MIM019: Warp and novaCTF tomograms are phase-flipped; the etomo tomogram is not.

FSC is reported as first and last 0.143 crossing: without CTF correction, all particles of a tomogram
share similar defocus, so the FSC dips at CTF zeros (MIM019: ~37, 26, 21 A) before the real limit.

## Conventions (see `src/py3sta/geometry.py`, `src/py3sta/io/particles.py`)

- Internal table = RELION 4 tomo particles: `rlnTomoName`, `rlnCoordinateX/Y/Z` in **unbinned
  tilt-series pixels**, `rlnOriginX/Y/ZAngst` (true centre = coord − origin/apix), `rlnAngleRot/Tilt/Psi`.
- Coordinates: 0 = centre of first voxel; binning changes are a pure scale (Warp convention).
  For IMOD `binvol`-binned tomograms use `--offset (bin-1)/2`.
- i3 trf matrix `M` maps tomogram → reference (`M @ normal = +z`), and
  `M = Rz(rot)·Ry(tilt)·Rz(psi) = A_relion.T` — checked against RELION's `Euler_angles2matrix`.
- i3 `dx,dy,dz`: true centre = `x + dx` (as in trf2star.py). **Not yet verified** whether i3 stores
  these shifts in the tomogram frame or the rotated motif frame.
- Dipole picks also set `rlnAngleTiltPrior/PsiPrior` (membrane normal) for RELION restraints.
