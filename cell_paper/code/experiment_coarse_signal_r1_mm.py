"""
Experiment: recovering a signal that lives in a coarse wavelet subspace at fine
oversampling r = m/N^2 = 1, 

Reported: dist(x_hat, x_c)/||x_c|| (global-phase aligned), mean +- std over
n_trials realisations, for both models.

Outputs:
  * coarse_signal_r1_mm_recon.png  
  * coarse_signal_r1_mm.npz        

Run:  py -3 experiment_coarse_signal_r1_mm.py        (PHASERET_BACKEND=cupy for GPU)
"""

import numpy as np

from phaseret_core import (
    load_cells3d, make_gaussian_A, pure_phase_signal, build_scales,
    coarsen_truth, upsample_replicate, spectral_init, gradient_descent,
    solve_multiscale, align_global_phase, rel_error, xp, asnumpy, backend_name,
    fig_path, data_path
)



CFG = dict(
    side=64,             # full-resolution side N
    h0=32,               # coarsest multiscale side; x_c lives on the N/2 grid
    ratio=1.0,           # fine-scale oversampling r = m / N^2
    n_trials=10,         # random realisations
    phase_scale=np.pi,
    iters=800,           # GD iters per scale; direct gets iters * n_scales
    n_power=150,         # power iterations for the MM spectral inits
    seed=0,
    fontsize=18,
    plot=True,
    save_npz="coarse_signal_r1_mm.npz"
)


def run(cfg=CFG):
    N, h0 = cfg["side"], cfg["h0"]
    n_fine = N * N
    hc = N // 2
    m = int(round(cfg["ratio"] * n_fine))
    scales = build_scales(N, h0)
    assert hc in scales
    iters_direct = cfg["iters"] * len(scales)

    x = pure_phase_signal(load_cells3d(N), cfg["phase_scale"])
    x_c = upsample_replicate(coarsen_truth(x, N, hc), hc, N)   # the SOLUTION

    print(f"Coarse-subspace signal at r={cfg['ratio']:g}, MM init in both solvers"
          f"  |  N={N} (n={n_fine}, m={m})  |  "
          f"delta_fine={m / n_fine:g}, delta_coarse={m / hc ** 2:g}  |  "
          f"backend={backend_name()}")

    T = cfg["n_trials"]
    err_d = np.zeros(T); err_m = np.zeros(T)
    xd_list, xm_list = [], []
    for t in range(T):
        ss = np.random.SeedSequence([cfg["seed"], t])
        rng_A, rng_d, rng_m = (np.random.default_rng(s) for s in ss.spawn(3))
        A = make_gaussian_A(m, n_fine, rng_A)
        y = xp.abs(A @ x_c) ** 2                 # x_c GENERATES the data

        x0 = spectral_init(A, y, n_power=cfg["n_power"], rng=rng_d)
        xd, _ = gradient_descent(A, y, x0, iters=iters_direct)
        xm, _ = solve_multiscale(A, y, N, scales, cfg["iters"], rng=rng_m, n_power=cfg["n_power"])

        err_d[t] = rel_error(xd, x_c)
        err_m[t] = rel_error(xm, x_c)
        xd_list.append(xd); xm_list.append(xm)
        print(f"  trial {t}:  dist/||.|| vs x_c   direct={err_d[t]:.3e}   "
              f"multiscale={err_m[t]:.3e}")

    print(f"\n  dist(x_hat, x_c)/||x_c||, mean +- std over {T} realisations:")
    print(f"    direct (MM init)    : {err_d.mean():.3e} +- {err_d.std():.3e}")
    print(f"    multiscale (MM init): {err_m.mean():.3e} +- {err_m.std():.3e}")

    med_t = int(np.argsort(err_m)[T // 2])
    if cfg.get("save_npz"):
        np.savez(data_path(cfg["save_npz"]),
                 x=asnumpy(x), x_c=asnumpy(x_c),
                 xd=asnumpy(xd_list[med_t]), xm=asnumpy(xm_list[med_t]),
                 err_d=err_d, err_m=err_m, med_t=med_t, N=N, init="mm")
        print(f"saved {cfg['save_npz']}")
    if cfg["plot"]:
        plot_recon(x_c, xd_list[med_t], xm_list[med_t], N, cfg["fontsize"])
    return err_d, err_m


def plot_recon(x_c, xd, xm, N, fontsize=18,
               fname="coarse_signal_r1_mm_recon.png"):
    """One row: downsampled gt (= the solution) | direct-MM | multiscale-MM."""
    import matplotlib.pyplot as plt
    panels = [
        (x_c, r"$\angle\mathbf{x}^{\star}_{c}$ (solution)"),
        (xd, "direct (MM init)"),
        (xm, "multiscale (MM init)"),
    ]
    fs = fontsize
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.9))
    im = None
    for ax, (z, title) in zip(np.atleast_1d(axes).reshape(-1), panels):
        im = ax.imshow(np.angle(asnumpy(align_global_phase(z, x_c))).reshape(N, N),
                       cmap="twilight", vmin=-np.pi, vmax=np.pi,
                       interpolation="nearest")
        ax.set_title(title, fontsize=fs); ax.axis("off")
    cb = fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), fraction=0.046,
                      pad=0.02, ticks=[-np.pi, 0, np.pi])
    cb.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=fs)
    fig.savefig(fig_path(fname), dpi=150, bbox_inches="tight")
    print(f"saved {fname}")


def replot(path="coarse_signal_r1_mm.npz", fontsize=18,
           fname="coarse_signal_r1_mm_recon.png"):
    """Re-render the row from stored reconstructions (no solve)."""
    d = np.load(data_path(path))
    plot_recon(d["x_c"], d["xd"], d["xm"], int(d["N"]), fontsize, fname)


if __name__ == "__main__":
    run()
