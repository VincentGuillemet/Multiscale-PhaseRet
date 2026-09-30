"""

Core library for experiment of "Oversampling Gains from Multiscale Resolution of Phase Retrieval".

Because |A x|^2 only determines x up to a global phase, all errors are reported
after best global-phase alignment.

Requires numpy, matplotlib and pywt; cupy is optional (GPU backend).
"""

import os

import numpy as np


###################################### Helper - backend ################################################


def asnumpy(a):
    if _cp is not None and isinstance(a, _cp.ndarray):
        return _cp.asnumpy(a)
    return np.asarray(a)


def to_device(a):
    return xp.asarray(a)


def backend_name():
    return "cupy(GPU)" if ON_GPU else "numpy(CPU)"




###################################### Backend ################################################

def _register_pip_cuda_dlls():
    """Windows only: make the CUDA runtime DLLs installed via pip wheels
    (nvidia-*-cu12 packages) visible to CuPy.  Python >= 3.8 does not search
    PATH for extension-module dependencies, so the wheel bin dirs must be
    registered explicitly.  No-op if the wheels are absent."""
    if os.name != "nt":
        return
    import glob
    import site
    roots = site.getsitepackages() + [site.getusersitepackages()]
    dirs = []
    for root in roots:
        for d in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            dirs.append(d)
            try:
                os.add_dll_directory(d)      # for python extension modules
            except OSError:
                pass
    if dirs:                                 # for native DLL -> DLL loads
        os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])


try:
    _register_pip_cuda_dlls()
    import cupy as _cp
except Exception:                                   # not installed
    _cp = None


def _select_backend():
    want = os.environ.get("PHASERET_BACKEND", "auto").lower()
    if want in ("numpy", "cpu"):
        return np, False
    if _cp is not None:
        try:
            if _cp.cuda.runtime.getDeviceCount() > 0:
                return _cp, True
        except Exception:
            pass
    if want in ("cupy", "gpu"):
        raise RuntimeError("PHASERET_BACKEND=cupy but CuPy/GPU is unavailable")
    return np, False


xp, ON_GPU = _select_backend()



###################################### Problem construction ################################################

def make_gaussian_A(m, n, rng):

    A = (rng.standard_normal((m, n))
         + 1j * rng.standard_normal((m, n))) / np.sqrt(2.0)
    return to_device(A)


def pure_phase_signal(image, phase_scale):
    """Pure-phase object"""
    image = asnumpy(image)
    return to_device(np.exp(1j * phase_scale * image.reshape(-1)))

###################################### Multiscale coarse->fine operators via an orthonormal WAVELET decomposition. ################################################

WAVELET = "haar"           # any ORTHOGONAL PyWavelets name ("haar", "sym4", ...)

_U_CACHE = {}


def scaling_filter(name=None):
    """Reconstruction low-pass (scaling) filter of an orthogonal wavelet, taken
    from PyWavelets so the taps are not transcribed by hand"""
    import pywt
    w = pywt.Wavelet(name or WAVELET)
    if not w.orthogonal:
        raise ValueError(f"{w.name!r} is not orthogonal; U^T U != I")
    return np.asarray(w.rec_lo, dtype=np.float64)


def _synth_1d(N, h):
    """1D L-level periodic orthonormal synthesis matrix (N x h): LL coeffs at
    length h (details zeroed) -> length-N signal."""
    g = scaling_filter()
    M = np.eye(h)
    s = h
    while s < N:                                    # one level h -> 2h at a time
        n = 2 * s
        S = np.zeros((n, s))                        # S[m,k] = g[(m-2k) mod n]
        for k in range(s):
            for i in range(len(g)):
                S[(2 * k + i) % n, k] += g[i]
        M = S @ M
        s = n
    return M


def _synthesis_matrix(N, h):
    """Orthonormal wavelet synthesis U (N^2 x h^2), cached
    on the active backend.  U @ c = full image from LL coeffs c; U^T = analysis."""
    key = (N, h, WAVELET)
    if key not in _U_CACHE:
        assert h & (h - 1) == 0 and N % h == 0, "N and h must be powers of two"
        M = _synth_1d(N, h)                         # N x h
        _U_CACHE[key] = to_device(np.kron(M, M))    # (N^2 x h^2)
    return _U_CACHE[key]


def coarsen_columns(A, N, h):
    if h == N:
        return A
    return A @ _synthesis_matrix(N, h)


def upsample_replicate(x, h, H):
    if h == H:
        return x
    return _synthesis_matrix(H, h) @ x


def coarsen_truth(x_true, N, h):
    if h == N:
        return x_true
    return _synthesis_matrix(N, h).T @ x_true


def build_scales(N, min_side):
    """Sides from coarse to fine: N, N/2, ... down to >= min_side (each | N)."""
    s, out = N, []
    while s >= min_side and N % s == 0:
        out.append(s)
        s //= 2
    return sorted(out)            # coarse -> fine




###################################### Evaluation ################################################


def align_global_phase(x_hat, x_ref):
    # x_hat is determined only up to a global phase; rotate it onto x_ref.
    ph = xp.exp(1j * xp.angle(xp.vdot(x_hat, x_ref)))
    return x_hat * ph


def rel_error(x_hat, x_ref):
    return float(xp.linalg.norm(align_global_phase(x_hat, x_ref) - x_ref)
                 / xp.linalg.norm(x_ref))


###################################### Objective and grad ################################################
def loss(A, x, y):
    #      f(x) = 1/(2m) * sum_i ( |(A x)_i| - sqrt(y_i) )^2
    w = A @ x
    return float(0.5 * xp.mean((xp.abs(w) - xp.sqrt(y)) ** 2))


def wirtinger_grad(A, x, y):
    #      grad = 1/(2m) * A^H [ (|w| - sqrt(y)) * w/|w| ]
    w = A @ x
    aw = xp.abs(w)
    phase = xp.where(aw > 1e-30, w / xp.where(aw > 1e-30, aw, 1.0), 0.0)
    r = (aw - xp.sqrt(y)) * phase
    return (A.conj().T @ r) / (2 * y.size)



###################################### Solver ################################################

def spectral_init(A, y, n_power=60, rng=None):
    """Leading eigenvector of D = (1/m) A^H diag(T(y)) A, scaled to best fit y,
    with the Mondelli-Montanari optimal preprocessing (Ann. Statist. 2019)

        T(t) = (t - 1) / (t + sqrt(delta) - 1),   t = y / mean(y), delta = m/n,

    which attains the weak-recovery threshold delta = 1 for complex Gaussian A
    (nontrivial correlation for any delta > 1).  T is clipped below at -5
    (t -> 0 sends T to -infinity as delta -> 1).  T can be negative, so D is
    indefinite: if plain power iteration locks onto the NEGATIVE spectral edge
    (Rayleigh quotient < 0), it is re-run on D + s I with s > |rho|.
    """
    rng = rng or np.random.default_rng(0)
    m, n = A.shape
    t = y / (xp.mean(y) + 1e-30)
    t_y = (t - 1.0) / (t + np.sqrt(m / n) - 1.0 + 1e-12)
    t_y = xp.maximum(t_y, -5.0)

    def apply_D(u):
        return A.conj().T @ (t_y * (A @ u)) / m

    v = to_device(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    v /= xp.linalg.norm(v)
    for _ in range(n_power):
        v = apply_D(v)
        v /= xp.linalg.norm(v) + 1e-30
    rho = float(xp.real(xp.vdot(v, apply_D(v))))
    if rho < 0.0:                       # converged to the negative edge: shift
        s = 1.5 * abs(rho)
        for _ in range(2 * n_power):
            v = apply_D(v) + s * v
            v /= xp.linalg.norm(v) + 1e-30
    # closed-form amplitude: minimize || c^2 |A v|^2 - y ||^2 over c^2
    u = xp.abs(A @ v) ** 2
    c2 = float((u @ y) / (u @ u + 1e-30))
    return float(np.sqrt(max(c2, 1e-30))) * v


def gradient_descent(A, y, x0, iters=400, eta0=1.0, c=1e-4, tol=0.0,
                     verbose=False):
    """Gradient descent on the amplitude loss with Armijo backtracking"""
    
    x = xp.asarray(x0).astype(xp.complex128).copy()
    eta = eta0
    f_curr = loss(A, x, y)
    history = [f_curr]
    for k in range(iters):
        g = wirtinger_grad(A, x, y)
        gn2 = float(xp.real(xp.vdot(g, g)))
        eta = min(eta * 2.0, 1e6)                 # try to grow the step
        while True:                               # Armijo backtracking
            xn = x - eta * g
            if loss(A, xn, y) <= f_curr - c * eta * gn2 or eta < 1e-14:
                break
            eta *= 0.5
        x_prev, x = x, xn
        f_curr = loss(A, x, y)
        history.append(f_curr)
        if verbose and (k % max(1, iters // 5) == 0 or k == iters - 1):
            print(f"      iter {k:4d}   loss {f_curr:.4e}   eta {eta:.2e}")
        if tol > 0.0:                             # stop once iterates converge
            rel_change = float(xp.linalg.norm(x - x_prev)
                               / (xp.linalg.norm(x) + 1e-30))
            if rel_change < tol:
                break
    return x, history


###################################### High-level solvers ################################################

def solve_direct(A, y, iters, rng=None, tol=0.0, n_power=60):

    x0 = spectral_init(A, y, n_power=n_power, rng=rng)
    return gradient_descent(A, y, x0, iters=iters, tol=tol)


def solve_multiscale(A, y, N, scales, iters, rng=None, verbose=False,
                     tol=0.0, n_power=60):

    x_coarse, prev_h = None, None
    per_scale = []
    for h in scales:
        A_h = coarsen_columns(A, N, h)
        if x_coarse is None:                           # coarsest: spectral init
            x0 = spectral_init(A_h, y, n_power=n_power, rng=rng)
            init = "spectral(mm)"
        else:                                          # replicate finer
            x0 = upsample_replicate(x_coarse, prev_h, h)
            init = f"upsample {prev_h}->{h}"
        x_h, hist = gradient_descent(A_h, y, x0, iters=iters,
                                     verbose=verbose, tol=tol)
        per_scale.append((h, x_h, hist, init))
        x_coarse, prev_h = x_h, h
    return x_coarse, per_scale




###################################### resizing  ################################################

def resize_bilinear(img, out_hw):
    H, W = img.shape
    oh, ow = out_hw
    yi = np.linspace(0, H - 1, oh)
    xi = np.linspace(0, W - 1, ow)
    y0 = np.floor(yi).astype(int); x0 = np.floor(xi).astype(int)
    y1 = np.minimum(y0 + 1, H - 1); x1 = np.minimum(x0 + 1, W - 1)
    wy = (yi - y0)[:, None]; wx = (xi - x0)[None, :]
    Ia = img[np.ix_(y0, x0)]; Ib = img[np.ix_(y0, x1)]
    Ic = img[np.ix_(y1, x0)]; Id = img[np.ix_(y1, x1)]
    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    return top * (1 - wy) + bot * wy


###################################### PATHS ################################################

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.environ.get(
    "PHASERET_FIGDIR", os.path.join(_HERE, os.pardir, "figures")))
DATA_DIR = _HERE


def fig_path(name):
    """Absolute path of an output figure inside FIG_DIR.
    An already-absolute name is passed through unchanged."""
    if os.path.isabs(name):
        return name
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)


def data_path(name):
    """Absolute path of a stored-array (.npz) file next to the code."""
    return name if os.path.isabs(name) else os.path.join(DATA_DIR, name)


_CELLS3D_PNG = os.path.join(FIG_DIR, "cells3d_nuclei_256.png")


def load_cells3d(side, path=_CELLS3D_PNG):
    """Return a side x side float image in [0, 1]: nuclei channel (z = 30) of
    the confocal fluorescence stack `cells3d` (Allen Institute for Cell
    Science), bundled next to the figures as a 256x256 PNG."""
    import matplotlib.image as mpimg
    img = np.asarray(mpimg.imread(path), dtype=np.float64)   # float RGBA in [0,1]
    return resize_bilinear(img[..., :3].mean(axis=2), (side, side))




