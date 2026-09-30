"""
Experiment: reconstruction quality vs fine-scale OVERSAMPLING RATIO r = m / N^2,
with the Mondelli-Montanari (MM) optimal spectral preprocessing in the
initialisation of both solvers.

Quality is the global-phase-invariant cosine similarity
    cos(theta) = |<x_hat, x>| / (||x_hat|| ||x||)   in [0, 1].

Outputs:
  * oversampling_curve_mm.png      two cosine curves (mean +/- std) vs r
  * oversampling_solutions_mm.npz  per-(r, realisation, method) reconstructions

Run:  py -3 experiment_oversampling_curve_mm.py      (PHASERET_BACKEND=cupy for GPU)
"""

import math

import numpy as np

import phaseret_core as C
from phaseret_core import (
    load_cells3d, pure_phase_signal, make_gaussian_A,
    solve_direct, solve_multiscale, build_scales, xp, asnumpy, backend_name,
    fig_path, data_path
)

# --------------------------------------------------------------------------- #
#  Hyperparameters (same as experiment_oversampling_curve.py + MM init)
# --------------------------------------------------------------------------- #
CFG = dict(
    side=32,                    # fine grid N x N (ground truth)
    min_side=8,                 # coarsest multiscale grid -> scales 8, 16, 32
    r_min=0.5, r_max=4.0, n_r=15,   # oversampling ratios r = m / N^2
    n_noise=10,                 # A realisations per r
    phase_scale=np.pi,
    iters=800,                  # MAX GD iters per scale (multiscale) / total*scales (direct)
    tol=1e-5,                   # stop when ||x-x_prev||/||x|| < tol
    n_power=150,                # power iterations for the MM spectral inits
    wavelet="sym4",             # coarse subspace for the multiscale solver
    seed=0,
    save="oversampling_curve_mm.png",
    save_sols="oversampling_solutions_mm.npz"
)


def cos_sim(x_hat, x):
    num = abs(complex(xp.vdot(x_hat, x)))
    return num / (float(xp.linalg.norm(x_hat)) * float(xp.linalg.norm(x)) + 1e-30)


def run(cfg=CFG):
    N, hc = cfg["side"], cfg["min_side"]
    n = N * N
    scales = build_scales(N, hc)                      # [8, 16, 32]
    rs = np.linspace(cfg["r_min"], cfg["r_max"], cfg["n_r"])
    C.WAVELET = cfg["wavelet"]

    x = pure_phase_signal(load_cells3d(N), cfg["phase_scale"])

    print(f"Oversampling curve, MM init  |  "
          f"N={N} (n={n})  |  scales={scales} "
          f"({cfg['wavelet']})  |  r in [{cfg['r_min']}, {cfg['r_max']}] x{cfg['n_r']} "
          f" |  {cfg['n_noise']} realisations  |  tol={cfg['tol']:g}  "
          f"|  backend={backend_name()}")

    R = cfg["n_noise"]
    cos_ms = np.zeros((cfg["n_r"], R))
    cos_dir = np.zeros((cfg["n_r"], R))
    sols_ms = np.zeros((cfg["n_r"], R, n), dtype=np.complex128)
    sols_dir = np.zeros((cfg["n_r"], R, n), dtype=np.complex128)
    for i, r in enumerate(rs):
        m = int(round(r * n))
        for j in range(R):
            ss = np.random.SeedSequence([cfg["seed"], i, j])
            rng_A, rng_d, rng_m = (np.random.default_rng(k) for k in ss.spawn(3))
            A = make_gaussian_A(m, n, rng_A)
            y = xp.abs(A @ x) ** 2

            xd, _ = solve_direct(A, y, iters=cfg["iters"] * len(scales), rng=rng_d,
                                 tol=cfg["tol"], n_power=cfg["n_power"])
            xm, _ = solve_multiscale(A, y, N, scales, iters=cfg["iters"], rng=rng_m,
                                     tol=cfg["tol"], n_power=cfg["n_power"])
            cos_dir[i, j] = cos_sim(xd, x)
            cos_ms[i, j] = cos_sim(xm, x)
            sols_dir[i, j] = asnumpy(xd)
            sols_ms[i, j] = asnumpy(xm)

        print(f"  r={r:4.2f} (m={m:5d})  cos  multiscale={cos_ms[i].mean():.3f}"
              f"  direct={cos_dir[i].mean():.3f}")

    if cfg["save"]:
        plot_curve(rs, cos_ms, cos_dir, cfg["save"])
    if cfg.get("save_sols"):
        np.savez_compressed(
            data_path(cfg['save_sols']), rs=rs, cos_ms=cos_ms, cos_dir=cos_dir,
            sols_ms=sols_ms, sols_dir=sols_dir, x_true=asnumpy(x),
            N=N, scales=np.array(scales), wavelet=cfg["wavelet"],
            seed=cfg["seed"], init="mm")
        print(f"saved {cfg['save_sols']}  "
              f"(sols_ms / sols_dir shape [n_r, n_noise, {n}] complex)")
    return rs, cos_ms, cos_dir


FONTSIZE = 13                     # same unified font size as omega_split_mm_cos


def plot_curve(rs, cos_ms, cos_dir, fname, fontsize=FONTSIZE):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not available, skipping plot)")
        return
    fs = fontsize
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for arr, label, color in [(cos_ms, "multiscale (MM init)", "tab:blue"),
                              (cos_dir, "direct (MM init)", "tab:red")]:
        mean, std = arr.mean(1), arr.std(1)
        ax.plot(rs, mean, "o-", color=color, label=label)
        ax.fill_between(rs, mean - std, mean + std, color=color, alpha=0.2)
    ax.axhline(1.0, color="0.6", ls=":", lw=1)
    ax.set_xlabel(r"$r$", fontsize=fs)
    ax.set_ylabel(r"$\cos\theta$", fontsize=fs)
    ax.set_ylim(0, 1.03)
    ax.tick_params(labelsize=fs)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fs)
    fig.tight_layout()
    fig.savefig(fig_path(fname), dpi=140)
    print(f"saved {fname}")


def replot(path="oversampling_solutions_mm.npz", fontsize=14,
           fname="oversampling_curve_mm.png"):
    """Reproduce the plot from saved solutions (no experiment rerun), with all
    individual realisations, means, and the random-vector baseline."""
    import matplotlib.pyplot as plt
    d = np.load(data_path(path))
    rs, x = d["rs"], d["x_true"]
    xn = np.linalg.norm(x)

    def cosines(sols):                               # (n_r, n_noise)
        num = np.abs(np.einsum("ijk,k->ij", sols, np.conj(x)))
        return num / (np.linalg.norm(sols, axis=2) * xn + 1e-30)

    c_ms, c_dir = cosines(d["sols_ms"]), cosines(d["sols_dir"])
    n = x.size
    floor = math.exp(math.lgamma(1.5) + math.lgamma(n) - math.lgamma(n + 0.5))

    fs = fontsize
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for c, label, color, marker in [(c_ms, "multiscale (MM init)", "tab:blue", "o"),
                                    (c_dir, "direct (MM init)", "tab:red", "s")]:
        ax.scatter(np.repeat(rs[:, None], c.shape[1], 1), c,
                   s=16, color=color, alpha=0.35, edgecolors="none", marker=marker)
        ax.plot(rs, c.mean(1), marker=marker, ls="-", color=color, label=label)
    ax.axhline(1.0, color="0.6", ls=":", lw=1)
    ax.axhline(floor, color="0.35", ls="--", lw=1.6,
               label="\U0001D53C" + fr"$(\cos\theta)\approx{floor:.3f}$")
    ax.set_xlabel(r"$r$", fontsize=fs)
    ax.set_ylabel(r"$\cos\theta$", fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.legend(fontsize=fs)
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path(fname), dpi=140)
    print(f"saved {fname}")


if __name__ == "__main__":
    run()

