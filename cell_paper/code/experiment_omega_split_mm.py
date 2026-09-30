"""
Experiment: omega split (coarse-only Wirtinger flow) with the Mondelli-Montanari

The recorded metric of interest is the coarse-direction cosine
and the plot shows it as a function of the off-subspace energy
fraction
    rho = ||omega * x_perp|| / ||s(omega)||                     in [0, 1],

Outputs:
  * omega_split_mm.npz       arrays: omegas, rho, err, corr, e_ratio, ...
  * omega_split_mm_cos.png   mean +/- std of cos(theta) vs rho

Run:  py -3 experiment_omega_split_mm.py             (PHASERET_BACKEND=cupy for GPU)
"""

import math

import numpy as np

from phaseret_core import (
    load_cells3d, pure_phase_signal, make_gaussian_A,
    coarsen_columns, coarsen_truth, upsample_replicate,
    spectral_init, gradient_descent, align_global_phase,
    xp, asnumpy, backend_name, WAVELET,
    fig_path, data_path
)

# --------------------------------------------------------------------------- #
#  Hyperparameters (same sweep as experiment_omega_split.py)
# --------------------------------------------------------------------------- #
CFG = dict(
    side=32,                    # fine side N  (fine grid 32x32)
    hc=16,                      # coarse side  (coarse grid 16x16)
    n_omega=20,                 # number of omega values, uniform in [0, 1]
    n_noise=10,                 # random realisations per omega
    coarse_oversampling=4.0,    # m = coarse_oversampling * hc^2  (>=4 => injective)
    phase_scale=np.pi,
    iters=800,                  # coarse Wirtinger-flow iterations
    n_power=150,                # power iterations for the MM spectral init
    seed=0,
    save="omega_split_mm.npz",
    plot=True
)


def run(cfg=CFG):
    N, hc = cfg["side"], cfg["hc"]
    n_fine, n_c = N * N, hc * hc
    m = int(round(cfg["coarse_oversampling"] * n_c))
    omegas = np.linspace(0.0, 1.0, cfg["n_omega"])

    x = pure_phase_signal(load_cells3d(N), cfg["phase_scale"])
    x_par_c = coarsen_truth(x, N, hc)                  # coarse coeffs  (U_c^T x)
    x_par = upsample_replicate(x_par_c, hc, N)         # coarse part    (U_c U_c^T x)
    x_perp = x - x_par                                 # detail part

    print(f"Omega split, MM init (coarse-only WF)  |  "
          f"N={N} -> hc={hc} "
          f"(n_c={n_c}, m={m}, coarse m/n_c={m/n_c:.1f})  |  wavelet={WAVELET}  "
          f"backend={backend_name()}")
    print(f"  coarse capture ||x_par||^2/||x||^2 = "
          f"{float(xp.linalg.norm(x_par) ** 2 / xp.linalg.norm(x) ** 2):.3f}")
    print(f"  sweep: {cfg['n_omega']} omega x {cfg['n_noise']} realisations "
          f"= {cfg['n_omega'] * cfg['n_noise']} coarse solves")

    E, R = cfg["n_omega"], cfg["n_noise"]
    err = np.zeros((E, R))
    corr = np.zeros((E, R))                            # the cosine cos(theta)
    e_ratio = np.zeros((E, R))
    loss_final = np.zeros((E, R))
    sig_norm = np.zeros(E)
    xpc_norm = float(xp.linalg.norm(x_par_c))
    perp_norm = float(xp.linalg.norm(x_perp))

    for i, w in enumerate(omegas):
        s = w * x_perp + (1.0 - w) * x_par
        snorm = float(xp.linalg.norm(s))
        sig_norm[i] = snorm
        target = (1.0 - w) * x_par_c
        for j in range(R):
            ss = np.random.SeedSequence([cfg["seed"], i, j])
            rng_A, rng_s = (np.random.default_rng(k) for k in ss.spawn(2))
            A = make_gaussian_A(m, n_fine, rng_A)
            y = xp.abs(A @ s) ** 2
            A_c = coarsen_columns(A, N, hc)            # A U_c   (m x n_c)

            x0 = spectral_init(A_c, y, n_power=cfg["n_power"], rng=rng_s)
            c_hat, hist = gradient_descent(A_c, y, x0, iters=cfg["iters"])

            d = float(xp.linalg.norm(align_global_phase(c_hat, target) - target))
            err[i, j] = d / snorm
            cn = float(xp.linalg.norm(c_hat))
            corr[i, j] = abs(complex(xp.vdot(c_hat, x_par_c))) / (cn * xpc_norm + 1e-30)
            e_ratio[i, j] = (cn / snorm) ** 2
            loss_final[i, j] = hist[-1]

        rho_i = w * perp_norm / snorm
        print(f"  omega={w:5.3f}  rho={rho_i:5.3f}  err mean={err[i].mean():.3e} "
              f"cos mean={corr[i].mean():.3f}   E-ratio mean={e_ratio[i].mean():.3f}")

    # off-subspace energy fraction rho(omega) = ||w x_perp|| / ||s||  (in [0,1])
    rho = omegas * perp_norm / sig_norm

    if cfg["save"]:
        np.savez(data_path(cfg["save"]), omegas=omegas, rho=rho, err=err, corr=corr,
                 e_ratio=e_ratio, loss_final=loss_final, sig_norm=sig_norm,
                 coarse_norm=xpc_norm, perp_norm=perp_norm,
                 N=N, hc=hc, m=m, wavelet=WAVELET, init="mm")
        print(f"saved {cfg['save']}")
    if cfg["plot"]:
        # exact chance floor of the cosine for a random complex-Gaussian c in C^{n_c}
        floor = math.exp(math.lgamma(1.5) + math.lgamma(n_c)
                         - math.lgamma(n_c + 0.5))
        plot_cos_vs_rho(rho, corr, chance_floor=floor)
    return rho, corr


FONTSIZE = 13


def plot_cos_vs_rho(rho, corr, chance_floor=None, fontsize=FONTSIZE,
                    fname="omega_split_mm_cos.png"):
    """Only the cosine curve, as a function of the off-subspace energy rho."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not available, skipping plot)")
        return
    fs = fontsize
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    mean = corr.mean(axis=1)
    std = corr.std(axis=1)
    ax.plot(rho, mean, "o-", label=r"$\cos(\theta)$  (MM init)")
    ax.fill_between(rho, mean - std, mean + std, alpha=0.2)
    if chance_floor is not None:
        ax.axhline(chance_floor, color="0.4", ls=":", lw=1.4,
                   label="\U0001D53C" + fr"$(\cos\theta)\approx{chance_floor:.3f}$")
    ax.set_xlabel(r"$\rho$  (off-subspace energy)", fontsize=fs)
    ax.set_ylabel(r"$\cos(\theta)$", fontsize=fs)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(labelsize=fs)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fs)
    fig.tight_layout(); fig.savefig(fig_path(fname), dpi=130)
    print(f"saved {fname}")


def replot(path="omega_split_mm.npz", fontsize=FONTSIZE,
           fname="omega_split_mm_cos.png"):
    """Reproduce the cos-vs-rho plot from the saved .npz (no experiment rerun)."""
    d = np.load(data_path(path))
    rho, corr, n_c = d["rho"], d["corr"], int(d["hc"]) ** 2
    floor = math.exp(math.lgamma(1.5) + math.lgamma(n_c) - math.lgamma(n_c + 0.5))
    plot_cos_vs_rho(rho, corr, chance_floor=floor, fontsize=fontsize, fname=fname)


if __name__ == "__main__":
    run()
