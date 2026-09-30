# Cell-image experiments (reproduction package)

Self-contained: everything needed to reproduce the figures of the paper on the
cell-nuclei phase object. No dependency on the rest of the repository.

## Layout

    cell_paper/
      code/      this folder: .py sources, stored results (.npz), this README
      figures/   all .png -- the generated figures AND the source image

Figures are always written to `figures/`, whatever the working directory:
`phaseret_core.fig_path()` resolves names against `FIG_DIR` (default
`../figures`, override with the `PHASERET_FIGDIR` environment variable) and
creates the folder on demand.  Stored arrays resolve through
`phaseret_core.data_path()` to sit next to the code.  So both

    py -3 code/experiment_phases_64.py        # from cell_paper/
    py -3 experiment_phases_64.py             # from cell_paper/code/

put the figure in the same place.

NOTE: `figures/cells3d_nuclei_256.png` is the INPUT image, not a generated
figure -- do not delete it when clearing outputs.

## Object

`figures/cells3d_nuclei_256.png` is the nuclei channel (z = 30) of the `cells3d`
confocal fluorescence stack, data provided by the Allen Institute for Cell
Science and distributed with scikit-image, bundled here as a 256x256 PNG so
that the code needs only numpy + matplotlib + PyWavelets (CuPy optional,
for GPU); scikit-image itself is NOT required to run anything.
The phase object is `x = exp(i * pi * g)` with `g` the image rescaled to [0, 1].

`load_cells3d(side)` is the only image loader; every experiment calls it
directly.

NOTE on the object's statistics: as a phase object the image is ~89% of its
energy in the constant (DC) mode, because ~83% of its pixels are dark
background.  That inflates cos(theta) relative to a full-contrast object, so
the reported cosines should be read as an easier regime than a full-contrast
phase map would give (the sym4 coarse-subspace capture is 0.926 here).

## Files

Core library: `phaseret_core.py` (backend selection, wavelet multiscale
operators, MM spectral init, the amplitude-flow gradient solver, and the
`fig_path` / `data_path` helpers that fix the output locations).
It is trimmed to exactly what these four experiments exercise: the
amplitude objective, the Mondelli-Montanari spectral initialisation, and
plain gradient descent with Armijo backtracking (no FISTA momentum or
adaptive restart -- all four experiments now use the same solver).
Wavelet taps are read from PyWavelets via `scaling_filter()` rather than
transcribed, so `WAVELET` accepts any orthogonal PyWavelets name
(`haar` and `sym4` are the ones the experiments select); biorthogonal
families are rejected, since the multiscale operators assume U^T U = I.

| experiment                              | outputs                                                    |
|-----------------------------------------|------------------------------------------------------------|
| `experiment_coarse_signal_r1_mm.py`     | `coarse_signal_r1_mm{.npz,_recon.png}`                     |
| `experiment_oversampling_curve_mm.py`   | `oversampling_curve_mm.png`, `oversampling_solutions_mm.npz` |
| `experiment_phases_64.py`               | `phases_64_mm.npz`, `phases_64_mm_combined.png` |
| `experiment_omega_split_mm.py`          | `omega_split_mm{.npz,_cos.png}`                            |

## Running

    py -3 experiment_phases_64.py                 # CPU
    PHASERET_BACKEND=cupy py -3 experiment_phases_64.py    # GPU

Approximate GPU runtimes: phases_64 ~20 min (both solvers), coarse_signal_r1
~10 min, omega_split ~10 min, oversampling_curve ~50 min.

Every figure can be re-rendered from the stored `.npz` WITHOUT re-solving:

    py -3 -c "import experiment_phases_64 as m; m.replot()"
    py -3 -c "import experiment_coarse_signal_r1_mm as m; m.replot()"
    py -3 -c "import experiment_omega_split_mm as m; m.replot()"
    py -3 -c "import experiment_oversampling_curve_mm as m; m.replot()"

`experiment_phases_64.py` runs BOTH solvers in one pass: per r it draws one
operator A and one measurement vector y, hands them to the direct and the
multiscale solver with equal total budgets, stores everything in a single
`phases_64_mm.npz`, and renders the combined 2 x 3 figure.  `CFG["row_labels"]` (default `("standard", "ours")`) and
`CFG["arrow_label"]` (default `"difficulties"`) retitle the combined figure.

## Note

`phaseret_core.py` is duplicated here and in `../archive/` so that both trees
run standalone.  This copy is the canonical one; if you edit it, copy it over.
