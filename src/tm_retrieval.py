"""
Transmission Matrix Retrieval
Project 2: Binary Amplitude Modulation – GS-based TM Recovery

Algorithm overview
──────────────────
For a scattering medium with N input modes and M output pixels the
forward model is:

    y_k  =  |T · x_k|²          (intensity, camera)
    x_k  ∈ {0, 1}^N             (binary DMD pattern)

Because the DMD modulates only amplitude (no phase control) the
Gerchberg-Saxton iteration cannot use the standard phase-retrieval trick
directly in the input plane.  Instead we work in a "lifted" space:

    1.  Estimate a complex-valued TM  T ∈ ℂ^{M×N}  from the intensity data
        using an iterative alternating-projection scheme:
        a. Fix T, update phases of estimated fields.
        b. Fix phases, update T via least-squares (pseudo-inverse).
    2.  Repeat until convergence (or max_iter).

DC-term handling
────────────────
Binary patterns have a non-zero mean → a DC (unmodulated) component leaks
into every output frame.  We model this as an extra "DC column" appended to
the input matrix, effectively learning  T_aug = [T | t_dc]  where t_dc is
the background field contribution.
"""

import numpy as np
from pathlib import Path
import json
import time
from typing import Tuple, Optional
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def normalise_rows(A: np.ndarray) -> np.ndarray:
    """Row-normalise a matrix (unit L2 norm per row)."""
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return A / norms


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pixel-wise Pearson correlation between two real images."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean();  b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def intensity(field: np.ndarray) -> np.ndarray:
    """Complex field → real intensity (|field|²)."""
    return np.abs(field) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Dataset:
    X_train: np.ndarray    # (K_train, N)   binary {0,1}
    Y_train: np.ndarray    # (K_train, M)   measured intensity, flattened
    X_test:  np.ndarray    # (K_test,  N)
    Y_test:  np.ndarray    # (K_test,  M)
    N: int                 # number of input modes
    M: int                 # number of output pixels
    config: dict


def load_dataset(data_dir: str,
                 output_roi: Optional[Tuple[int,int,int,int]] = None
                 ) -> Dataset:
    """
    Load saved acquisition data.

    output_roi : (r0, r1, c0, c1) crop of camera frame to use as output.
                 None → use full frame.
    """
    p = Path(data_dir)
    with open(p / "config.json") as f:
        cfg = json.load(f)

    X_train = np.load(p / "train_inputs.npy").astype(np.float32)   # (K, N)
    Y_train_raw = np.load(p / "train_outputs.npy").astype(np.float32)  # (K, H, W)
    X_test  = np.load(p / "test_inputs.npy").astype(np.float32)
    Y_test_raw  = np.load(p / "test_outputs.npy").astype(np.float32)

    if output_roi is not None:
        r0, r1, c0, c1 = output_roi
        Y_train_raw = Y_train_raw[:, r0:r1, c0:c1]
        Y_test_raw  = Y_test_raw[:,  r0:r1, c0:c1]

    # Flatten spatial dims → (K, M)
    K_tr = Y_train_raw.shape[0]
    M    = Y_train_raw[0].size
    Y_train = Y_train_raw.reshape(K_tr, M)
    Y_test  = Y_test_raw.reshape(X_test.shape[0], M)

    N = X_train.shape[1]
    return Dataset(X_train, Y_train, X_test, Y_test, N, M, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# GS-BASED TM RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

class GSTMRetriever:
    """
    Iterative TM retrieval using alternating projections
    (Gerchberg-Saxton-style) adapted for binary amplitude modulation.

    Parameters
    ----------
    N           : number of input modes (DMD macropixels)
    M           : number of output pixels (camera ROI)
    max_iter    : GS iterations
    tol         : stop if relative change in T < tol
    reg         : Tikhonov regularisation for the LS step
    use_dc      : append DC column to model the unmodulated background
    verbose     : print progress
    """

    def __init__(self,
                 N: int, M: int,
                 max_iter: int = 50,
                 tol: float   = 1e-4,
                 reg: float   = 1e-3,
                 use_dc: bool = True,
                 verbose: bool = True):
        self.N        = N
        self.M        = M
        self.max_iter = max_iter
        self.tol      = tol
        self.reg      = reg
        self.use_dc   = use_dc
        self.verbose  = verbose

        self.T: Optional[np.ndarray] = None   # retrieved TM  (M, N[+1])
        self.history = []

    # ── Initialisation ────────────────────────────────────────────────────────
    def _build_input_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        X : (K, N)  binary {0, 1}
        Returns X_aug : (K, N+1) if use_dc else (K, N)
        The extra column is all-ones (DC term).
        """
        if self.use_dc:
            K = X.shape[0]
            return np.hstack([X, np.ones((K, 1), dtype=X.dtype)])
        return X

    def _n_aug(self) -> int:
        return self.N + 1 if self.use_dc else self.N

    # ── Least-squares update of T (fix phases) ────────────────────────────────
    def _ls_update_T(self,
                     X_aug: np.ndarray,
                     E_est: np.ndarray) -> np.ndarray:
        """
        Solve   X_aug @ T^H  ≈  E_est   in LS sense.

        X_aug  : (K, N_aug)
        E_est  : (K, M)   complex estimated fields
        T      : (M, N_aug)

        Tikhonov:  T^H = (X^H X + λI)^{-1} X^H E
        """
        K, N_aug = X_aug.shape
        # (N_aug, N_aug)
        A = X_aug.T @ X_aug + self.reg * np.eye(N_aug, dtype=np.complex128)
        B = X_aug.T @ E_est         # (N_aug, M)
        T_H = np.linalg.solve(A, B)  # (N_aug, M)
        return T_H.T                  # (M, N_aug)

    # ── GS phase update (fix T) ───────────────────────────────────────────────
    def _gs_phase_update(self,
                         T: np.ndarray,
                         X_aug: np.ndarray,
                         Y: np.ndarray) -> np.ndarray:
        """
        For each measurement k, estimate the complex output field:
            E_k_pred = T @ x_k              (predicted field)
            E_k_est  = sqrt(Y_k) * exp(i·∠E_k_pred)   (GS projection)

        T      : (M, N_aug)
        X_aug  : (K, N_aug)
        Y      : (K, M)   measured intensities

        Returns E_est : (K, M) complex
        """
        E_pred = (X_aug @ T.T)           # (K, M)
        phase  = np.angle(E_pred)        # (K, M)
        amp    = np.sqrt(np.maximum(Y, 0))  # measured amplitude
        return amp * np.exp(1j * phase)  # (K, M)

    # ── Main fit ──────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "GSTMRetriever":
        """
        X : (K, N)  binary input patterns  {0, 1}
        Y : (K, M)  measured intensity at output
        """
        K = X.shape[0]
        X_aug = self._build_input_matrix(X).astype(np.complex128)
        Y     = Y.astype(np.float64)

        # ── Init T with random phases, amplitude from mean output ─────────────
        rng = np.random.default_rng(0)
        amp_init = np.sqrt(np.maximum(Y.mean(axis=0), 0))  # (M,)
        N_aug    = self._n_aug()
        T = (np.tile(amp_init[:, None], (1, N_aug)) / np.sqrt(N_aug) *
             np.exp(1j * 2 * np.pi * rng.random((self.M, N_aug))))

        t0 = time.time()
        if self.verbose:
            print(f"\n[GS-TM] K={K}  N={self.N}  M={self.M}  "
                  f"N_aug={N_aug}  DC={'yes' if self.use_dc else 'no'}")
            print(f"{'Iter':>5}  {'Residual':>12}  {'ΔT/T':>10}  {'Time':>8}")
            print("─" * 45)

        for it in range(self.max_iter):
            T_old = T.copy()

            # Step 1 – GS phase projection
            E_est = self._gs_phase_update(T, X_aug, Y)

            # Step 2 – LS update of T
            T = self._ls_update_T(X_aug, E_est)

            # Diagnostics
            Y_pred  = intensity(X_aug @ T.T)          # (K, M)
            residual = float(np.mean((Y_pred - Y) ** 2))

            dT = np.linalg.norm(T - T_old) / (np.linalg.norm(T_old) + 1e-30)
            self.history.append({"iter": it, "residual": residual, "dT": dT})

            if self.verbose and (it % 5 == 0 or it < 3):
                print(f"{it+1:>5}  {residual:>12.6f}  {dT:>10.2e}  "
                      f"{time.time()-t0:>7.1f}s")

            if dT < self.tol:
                if self.verbose:
                    print(f"  Converged at iter {it+1}  (ΔT/T={dT:.2e} < {self.tol})")
                break

        self.T = T
        if self.verbose:
            print(f"\n[GS-TM] Done. Final MSE residual: {residual:.6f}")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict_field(self, X: np.ndarray) -> np.ndarray:
        """Predict complex output field.  Returns (K, M) complex array."""
        assert self.T is not None, "Call fit() first."
        X_aug = self._build_input_matrix(X).astype(np.complex128)
        return X_aug @ self.T.T

    def predict_intensity(self, X: np.ndarray) -> np.ndarray:
        """Predict intensity pattern.  Returns (K, M) float array."""
        return intensity(self.predict_field(X))

    # ── Evaluation ────────────────────────────────────────────────────────────
    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """
        Compute accuracy metrics on a test set.

        Returns dict with:
          mse        – mean squared error
          pearson    – mean Pearson correlation per output image
          psnr_db    – peak signal-to-noise ratio
        """
        Y_pred = self.predict_intensity(X)
        mse  = float(np.mean((Y_pred - Y) ** 2))
        psnr = float(10 * np.log10(Y.max() ** 2 / (mse + 1e-30)))
        cors = [pearson_correlation(Y_pred[k], Y[k]) for k in range(len(X))]
        return {"mse": mse, "pearson_mean": float(np.mean(cors)),
                "pearson_std":  float(np.std(cors)),  "psnr_db": psnr}

    # ── Save / load ───────────────────────────────────────────────────────────
    def save(self, path: str):
        np.save(path, self.T)
        print(f"[GS-TM] TM saved → {path}")

    def load(self, path: str):
        self.T = np.load(path).astype(np.complex128)
        print(f"[GS-TM] TM loaded from {path}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRIC STUDY  –  oversampling ratio sweep
# ─────────────────────────────────────────────────────────────────────────────

def parametric_study(dataset: Dataset,
                     ratios: Tuple[float, ...] = (1.0, 2.0, 4.0),
                     max_iter: int = 30,
                     output_roi: Optional[Tuple[int,int,int,int]] = None
                     ) -> dict:
    """
    Fix N (input modes) and vary K (number of training patterns).
    K = ratio × N.

    Returns a dict mapping ratio → evaluation metrics.
    """
    N  = dataset.N
    results = {}
    print("\n" + "═" * 55)
    print(f"  Parametric Study: oversampling ratios = {ratios}")
    print(f"  N (input modes) = {N}")
    print("═" * 55)

    for r in ratios:
        K = int(r * N)
        K = min(K, len(dataset.X_train))
        print(f"\n▶ Ratio {r:.1f}× →  K = {K} patterns")

        X_tr = dataset.X_train[:K]
        Y_tr = dataset.Y_train[:K]

        retriever = GSTMRetriever(N=N, M=dataset.M,
                                  max_iter=max_iter, verbose=True)
        retriever.fit(X_tr, Y_tr)
        metrics = retriever.evaluate(dataset.X_test, dataset.Y_test)

        print(f"   MSE      = {metrics['mse']:.5f}")
        print(f"   Pearson  = {metrics['pearson_mean']:.4f} "
              f"± {metrics['pearson_std']:.4f}")
        print(f"   PSNR     = {metrics['psnr_db']:.2f} dB")

        results[r] = {"K": K, "metrics": metrics,
                      "history": retriever.history}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# BONUS: PSF ENGINEERING via binary pattern optimisation
# ─────────────────────────────────────────────────────────────────────────────

def psf_engineer(retriever: GSTMRetriever,
                 target: np.ndarray,
                 n_iter: int = 200,
                 method: str = "greedy") -> np.ndarray:
    """
    Find a binary DMD pattern x ∈ {0,1}^N  that shapes the output
    intensity T·x to match the target spatial mode.

    Parameters
    ----------
    retriever : fitted GSTMRetriever
    target    : (M,) desired intensity profile (e.g. donut, Gaussian)
    n_iter    : optimisation iterations
    method    : 'greedy'  – bit-flip hill-climbing (simple, robust)
                'gs_bin'  – binarised GS (faster but less accurate)

    Returns
    -------
    x_opt : (N,) binary {0, 1} optimal DMD pattern
    """
    T    = retriever.T                          # (M, N[+1])
    N    = retriever.N
    use_dc = retriever.use_dc

    # Strip DC column for optimisation (DC is always present)
    T_no_dc = T[:, :N]  # (M, N)

    target = target / (target.max() + 1e-30)   # normalise

    rng = np.random.default_rng(42)
    x = rng.integers(0, 2, size=N).astype(np.float64)

    def score(xv):
        field = T_no_dc @ xv                        # (M,) complex
        pred  = intensity(field)
        pred  = pred / (pred.max() + 1e-30)
        return pearson_correlation(pred, target)

    if method == "greedy":
        # Stochastic bit-flip hill climbing
        best_score = score(x)
        for it in range(n_iter * N):
            i = rng.integers(N)
            x[i] = 1 - x[i]              # flip
            s = score(x)
            if s > best_score:
                best_score = s
            else:
                x[i] = 1 - x[i]          # revert
            if (it + 1) % (10 * N) == 0:
                print(f"  [PSF-greedy] iter {it+1}  Pearson={best_score:.4f}")

    elif method == "gs_bin":
        # GS-style: alternating between field and target constraints
        phase = 2 * np.pi * rng.random(N)
        for it in range(n_iter):
            # Forward: compute output field
            field_out = T_no_dc @ (x * np.exp(1j * phase))    # (M,)
            # Replace amplitude with sqrt(target), keep phase
            field_out = np.sqrt(target) * np.exp(1j * np.angle(field_out))
            # Backward: project back to input
            field_in  = T_no_dc.conj().T @ field_out           # (N,)
            # Binarise by sign of real part
            x     = (np.real(field_in) > 0).astype(np.float64)
            phase = np.angle(field_in)
            if (it + 1) % 20 == 0:
                print(f"  [PSF-GS-bin] iter {it+1}  "
                      f"Pearson={score(x):.4f}")
    else:
        raise ValueError(f"Unknown method: {method}")

    return x.astype(np.uint8)


def make_donut_target(M_side: int) -> np.ndarray:
    """Create a donut (annular) intensity pattern for PSF engineering."""
    cx = cy = M_side // 2
    y, x = np.ogrid[:M_side, :M_side]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    inner, outer = M_side * 0.15, M_side * 0.35
    donut = np.where((r >= inner) & (r <= outer), 1.0, 0.0)
    return donut.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO  (runs on synthetic data when called directly)
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_dataset(N: int = 64, M: int = 256,
                             K_train: int = 192, K_test: int = 32):
    """Generate fake data for offline unit-testing."""
    rng = np.random.default_rng(0)
    T_true = (rng.standard_normal((M, N)) +
              1j * rng.standard_normal((M, N))) / np.sqrt(N)

    X_train = rng.integers(0, 2, (K_train, N)).astype(np.float32)
    X_test  = rng.integers(0, 2, (K_test,  N)).astype(np.float32)

    def sim(X):
        E = X.astype(np.complex128) @ T_true.T
        Y = intensity(E).astype(np.float32)
        # Add ~2% shot noise
        Y += 0.02 * Y.max() * rng.standard_normal(Y.shape).astype(np.float32)
        return np.maximum(Y, 0)

    return Dataset(
        X_train=X_train, Y_train=sim(X_train),
        X_test=X_test,   Y_test=sim(X_test),
        N=N, M=M, config={}
    ), T_true


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("  GS-TM Retrieval Demo (synthetic data)")
    print("=" * 60)

    # ── Synthetic dataset ──
    N, M = 64, 256
    ds, T_true = _make_synthetic_dataset(N=N, M=M, K_train=4*N, K_test=32)

    # ── Parametric study ──
    results = parametric_study(ds, ratios=(1.0, 2.0, 4.0), max_iter=40)

    # Plot Pearson vs ratio
    ratios = sorted(results.keys())
    pearsons = [results[r]["metrics"]["pearson_mean"] for r in ratios]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([f"{r:.0f}×N" for r in ratios], pearsons,
                 "o-", color="#2196F3", linewidth=2, markersize=8)
    axes[0].set_title("TM Prediction Accuracy vs Oversampling Ratio",
                      fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Training patterns (ratio × N)")
    axes[0].set_ylabel("Mean Pearson Correlation")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)

    # Convergence of best model
    best_ratio = max(ratios)
    hist = results[best_ratio]["history"]
    iters = [h["iter"] for h in hist]
    resids = [h["residual"] for h in hist]
    axes[1].semilogy(iters, resids, color="#E91E63", linewidth=2)
    axes[1].set_title(f"GS Convergence  ({best_ratio:.0f}×N patterns)",
                      fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("MSE Residual (log)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/tm_study.png", dpi=150)
    print("\n[Plot] Saved → tm_study.png")

    # ── PSF Engineering demo ──
    print("\n[PSF] Retrieving TM at 4× oversampling for donut shaping …")
    retriever = GSTMRetriever(N=N, M=M, max_iter=50, verbose=False)
    retriever.fit(ds.X_train, ds.Y_train)

    M_side = int(np.sqrt(M))
    target = make_donut_target(M_side).ravel()
    x_opt  = psf_engineer(retriever, target, n_iter=100, method="gs_bin")
    pred   = intensity(retriever.predict_field(x_opt[None]))[0]

    fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4))
    axes2[0].imshow(target.reshape(M_side, M_side), cmap="inferno")
    axes2[0].set_title("Target: Donut", fontweight="bold")
    axes2[0].axis("off")
    axes2[1].imshow(pred.reshape(M_side, M_side), cmap="inferno")
    axes2[1].set_title(f"DMD Output (Pearson={pearson_correlation(pred, target):.3f})",
                       fontweight="bold")
    axes2[1].axis("off")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/psf_donut.png", dpi=150)
    print("[Plot] Saved → psf_donut.png")
