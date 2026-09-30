# Cell-image experiments

Reproduces the paper's figures for the cell-nuclei phase object.
Self-contained; needs numpy, matplotlib and pywt (cupy optional, for GPU).

    code/      sources, stored results (.npz)
    figures/   generated figures + the source image

## Running

    py -3 experiment_phases_64.py                           # CPU
    PHASERET_BACKEND=cupy py -3 experiment_phases_64.py     # GPU

| experiment                            | figure                          | GPU   |
|---------------------------------------|---------------------------------|-------|
| `experiment_phases_64.py`             | `phases_64_mm_combined.png`     | ~20 m |
| `experiment_coarse_signal_r1_mm.py`   | `coarse_signal_r1_mm_recon.png` | ~10 m |
| `experiment_omega_split_mm.py`        | `omega_split_mm_cos.png`        | ~10 m |
| `experiment_oversampling_curve_mm.py` | `oversampling_curve_mm.png`     | ~50 m |

Each also stores a `.npz` beside the code, from which its figure re-renders
without re-solving:

    py -3 -c "import experiment_phases_64 as m; m.replot()"

## Object

`figures/cells3d_nuclei_256.png` is the nuclei channel (z = 30) of the `cells3d`
confocal fluorescence stack, data provided by the Allen Institute for Cell
Science and distributed with scikit-image.  It is an INPUT, not a generated
figure.  The phase object is `x = exp(i*pi*g)` with `g` the image in [0, 1].