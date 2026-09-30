"""
Experiment: phase reconstructions at r = 1, 2, 4 for the 64x64 pure-phase
object, comparing the DIRECT ("standard") and MULTISCALE ("ours") Wirtinger
flows, both with the Mondelli-Montanari (MM) optimal spectral initialisation.

Outputs (figures land in ../figures, arrays next to this file):
  * phases_64_mm.npz            both solvers' reconstructions + cosines
  * phases_64_mm_combined.png   2 x 3: "standard" over "ours", with the
                                right-to-left "difficulties" arrow

Run:  py -3 experiment_phases_64.py            (PHASERET_BACKEND=cupy for GPU)
Re-render the figure without solving:
      py -3 -c "import experiment_phases_64 as m; m.replot()"
"""

import numpy as np

import phaseret_core as C
from phaseret_core import (
    load_cells3d, pure_phase_signal, make_gaussian_A, solve_direct,
    solve_multiscale, build_scales, fig_path, data_path,
    xp, asnumpy, backend_name
)

CFG = dict(
    side=64,                    # fine grid N x N
    min_side=8,                 # coarsest grid -> scales 8, 16, 32, 64
    rs=(1.0, 2.0, 4.0),         # oversampling ratios to show
    n_noise=1,                  # measurement realisations per r
    phase_scale=np.pi,
    iters=1000,                 # MAX GD iters per scale; direct gets iters * n_scales
    tol=1e-5,                   # stop once ||x-x_prev||/||x|| < tol
    n_power=150,                # power iterations for the MM spectral init
    wavelet="sym4",             # coarse subspace for the multiscale solver
    seed=0,
    fontsize=18,
    row_labels=("standard", "ours"),
    arrow_label="difficulties",
    save_npz="phases_64_mm.npz",
    save_combined="phases_64_mm_combined.png"
)


def cos_sim(x_hat, x):
    num = abs(complex(xp.vdot(x_hat, x)))
    return num / (float(xp.linalg.norm(x_hat)) * float(xp.linalg.norm(x)) + 1e-30)


# --------------------------------------------------------------------------- #
#  Experiment
# --------------------------------------------------------------------------- #
def run(cfg=CFG):
    N, hc = cfg["side"], cfg["min_side"]
    n = N * N
    scales = build_scales(N, hc)                      # [8, 16, 32, 64]
    iters_direct = cfg["iters"] * len(scales)         # equal total budget
    C.WAVELET = cfg["wavelet"]
    x = pure_phase_signal(load_cells3d(N), cfg["phase_scale"])

    print(f"Phases, MM init, direct vs multiscale  |  "
          f"N={N} (n={n})  |  scales={scales} ({cfg['wavelet']})  |  "
          f"r={cfg['rs']}  |  budget={iters_direct} iters both  |  "
          f"tol={cfg['tol']:g}  |  backend={backend_name()}")

    rs = np.array([float(r) for r in cfg["rs"]])
    sols_ms, sols_dir, cos_ms, cos_dir = [], [], [], []
    for i, r in enumerate(rs):
        m = int(round(r * n))
        trials = []
        for j in range(cfg["n_noise"]):
            # Same A and y for both solvers; each solver gets an identical but
            # independent init stream (reproduces the two former scripts).
            ss = np.random.SeedSequence([cfg["seed"], i, j])
            child_A, child_solver = ss.spawn(2)
            A = make_gaussian_A(m, n, np.random.default_rng(child_A))
            y = xp.abs(A @ x) ** 2

            xd, _ = solve_direct(A, y, iters=iters_direct,
                                 rng=np.random.default_rng(child_solver),
                                 tol=cfg["tol"],
                                 n_power=cfg["n_power"])
            xm, _ = solve_multiscale(A, y, N, scales, iters=cfg["iters"],
                                     rng=np.random.default_rng(child_solver),
                                     tol=cfg["tol"],
                                     n_power=cfg["n_power"])
            trials.append((xd, cos_sim(xd, x), xm, cos_sim(xm, x)))

        # representative trial = median multiscale cosine
        j_med = int(np.argsort([t[3] for t in trials])[len(trials) // 2])
        xd, cd, xm, cm = trials[j_med]
        sols_dir.append(asnumpy(xd)); cos_dir.append(cd)
        sols_ms.append(asnumpy(xm));  cos_ms.append(cm)
        print(f"  r={r:4.2f} (m={m:6d})  cos  direct={cd:.4f}  multiscale={cm:.4f}")

    sols_ms, sols_dir = np.stack(sols_ms), np.stack(sols_dir)
    cos_ms, cos_dir = np.array(cos_ms), np.array(cos_dir)

    if cfg.get("save_npz"):
        np.savez(data_path(cfg["save_npz"]),
                 x_true=asnumpy(x), rs=rs,
                 sols_ms=sols_ms, sols_dir=sols_dir,
                 cos_ms=cos_ms, cos_dir=cos_dir,
                 N=N, scales=np.array(scales), wavelet=cfg["wavelet"],
                 init="mm", iters=iters_direct)
        print(f"saved {cfg['save_npz']}")

    x_np = asnumpy(x)
    plot_combined(x_np, rs, sols_dir, cos_dir, sols_ms, cos_ms, N,
                  cfg["fontsize"], cfg["row_labels"], cfg["arrow_label"],
                  cfg["save_combined"])
    return rs, cos_dir, cos_ms


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def _aligner(x):
    """Rotate a reconstruction onto the ground truth's global phase."""
    def align(v):
        return v * np.exp(1j * np.angle(np.vdot(v, x)))
    return align


def plot_combined(x, rs, sols_dir, cos_dir, sols_ms, cos_ms, N, fontsize=18,
                  row_labels=("standard", "ours"), arrow_label="difficulties",
                  fname="phases_64_mm_combined.png"):
    """2 x k: direct on top, multiscale below, same A / y per column.  Row names
    are written vertically down the left margin, and a right-to-left arrow above
    the grid marks the direction of increasing difficulty (r decreasing)."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
    except Exception:
        print("(matplotlib not available, skipping plot)")
        return
    align = _aligner(x)
    rows = [(row_labels[0], sols_dir, cos_dir), (row_labels[1], sols_ms, cos_ms)]
    fs = fontsize
    k = len(rs)
    fig, axes = plt.subplots(2, k, figsize=(3.4 * k, 3.9 * 2))
    axes = np.asarray(axes).reshape(2, k)      # keep 2-D when k == 1
    im = None
    for ax_row, (label, sols, cos) in zip(axes, rows):
        for col, (ax, r, sol, c) in enumerate(zip(ax_row, rs, sols, cos)):
            im = ax.imshow(np.angle(align(sol)).reshape(N, N), cmap="twilight",
                           vmin=-np.pi, vmax=np.pi, interpolation="nearest")
            ax.set_title(f"$r={float(r):.0f}$\n"
                         + r"$\cos\theta=$" + f"{float(c):.3f}", fontsize=fs)
            # frame off, but keep the axis alive so the ylabel still renders
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if col == 0:                       # vertical row name in the margin
                ax.set_ylabel(label, fontsize=fs + 2, labelpad=12)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.02,
                      ticks=[-np.pi, 0, np.pi])
    cb.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=fs)

    fig.canvas.draw()
    p_left, p_right = axes[0, 0].get_position(), axes[0, -1].get_position()
    y = p_left.y1 + 0.085                      # clear of the two-line titles
    fig.add_artist(FancyArrowPatch(
        (p_right.x1, y), (p_left.x0, y),       # tail right -> head left
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=26,
        lw=2.2, color="0.15", shrinkA=0, shrinkB=0))
    fig.text(0.5 * (p_left.x0 + p_right.x1), y + 0.012, arrow_label,
             ha="center", va="bottom", fontsize=fs + 2)
    fig.savefig(fig_path(fname), dpi=150, bbox_inches="tight")
    print(f"saved {fname}")


def replot(path=None, cfg=CFG):
    """Re-render the combined figure from the stored arrays (no solving)."""
    d = np.load(data_path(path or cfg["save_npz"]))
    x, rs, N, fs = d["x_true"], d["rs"], int(d["N"]), cfg["fontsize"]
    plot_combined(x, rs, d["sols_dir"], d["cos_dir"], d["sols_ms"], d["cos_ms"],
                  N, fs, cfg["row_labels"], cfg["arrow_label"],
                  cfg["save_combined"])


if __name__ == "__main__":
    run()
