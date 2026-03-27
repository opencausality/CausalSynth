"""Statistical similarity metrics between real and synthetic data.

Provides:
- Per-column KS (Kolmogorov–Smirnov) tests: compare marginal distributions.
- Maximum Mean Discrepancy (MMD): compare multivariate distributions.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("causalsynth.validation.statistical")


def compute_ks_tests(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> dict[str, float]:
    """Compute a two-sample KS test for each numeric column.

    A high p-value (close to 1.0) means the real and synthetic distributions
    are statistically indistinguishable for that column.

    Args:
        real: Original real DataFrame.
        synthetic: Synthetic DataFrame.

    Returns:
        Dictionary mapping column name -> KS test p-value.
        Columns not present in both DataFrames are skipped.
    """
    results: dict[str, float] = {}

    common_cols = [
        c
        for c in real.columns
        if c in synthetic.columns and pd.api.types.is_numeric_dtype(real[c])
    ]

    for col in common_cols:
        real_vals = real[col].dropna().values
        synth_vals = synthetic[col].dropna().values

        if len(real_vals) == 0 or len(synth_vals) == 0:
            logger.warning("Column '%s' is empty; skipping KS test.", col)
            continue

        ks_stat, p_value = stats.ks_2samp(real_vals, synth_vals)
        results[col] = float(p_value)

        logger.debug(
            "KS test '%s': statistic=%.4f, p=%.4f",
            col,
            ks_stat,
            p_value,
        )

    return results


def compute_mmd(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    kernel: str = "rbf",
    gamma: float | None = None,
) -> float:
    """Compute Maximum Mean Discrepancy between real and synthetic data.

    MMD measures the distance between two distributions in a Reproducing
    Kernel Hilbert Space.  Lower MMD = more similar distributions.

    Uses RBF (Gaussian) kernel approximation.

    MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    where x ~ P_real, y ~ P_synthetic.

    Args:
        real: Real data DataFrame (numeric columns only).
        synthetic: Synthetic DataFrame (numeric columns only).
        kernel: Only "rbf" is currently supported.
        gamma: RBF kernel bandwidth parameter. If None, uses the median
               heuristic: gamma = 1 / (2 * median_pairwise_dist^2).

    Returns:
        Non-negative MMD score (lower = more similar).
    """
    if kernel != "rbf":
        raise ValueError(f"Unsupported kernel '{kernel}'. Only 'rbf' is supported.")

    # Select common numeric columns
    common_cols = [
        c
        for c in real.columns
        if c in synthetic.columns and pd.api.types.is_numeric_dtype(real[c])
    ]

    if not common_cols:
        logger.warning("No common numeric columns for MMD computation.")
        return float("nan")

    X = real[common_cols].dropna().values.astype(float)
    Y = synthetic[common_cols].dropna().values.astype(float)

    # Subsample for speed if datasets are large
    max_samples = 500
    if len(X) > max_samples:
        idx = np.random.default_rng(0).choice(len(X), max_samples, replace=False)
        X = X[idx]
    if len(Y) > max_samples:
        idx = np.random.default_rng(0).choice(len(Y), max_samples, replace=False)
        Y = Y[idx]

    # Standardise columns
    col_stds = X.std(axis=0)
    col_stds[col_stds < 1e-8] = 1.0
    X = (X - X.mean(axis=0)) / col_stds
    Y = (Y - Y.mean(axis=0)) / col_stds

    # Median heuristic for gamma
    if gamma is None:
        # Compute pairwise distances on a subsample
        n_sub = min(200, len(X), len(Y))
        X_sub = X[:n_sub]
        Y_sub = Y[:n_sub]
        Z = np.vstack([X_sub, Y_sub])
        sq_dists = _pairwise_sq_distances(Z, Z)
        # Use lower triangular (excluding diagonal)
        tril_vals = sq_dists[np.tril_indices_from(sq_dists, k=-1)]
        median_sq = float(np.median(tril_vals[tril_vals > 0])) if len(tril_vals) > 0 else 1.0
        gamma = 1.0 / (2.0 * max(median_sq, 1e-8))

    # Compute kernel matrices
    K_xx = _rbf_kernel(X, X, gamma)
    K_yy = _rbf_kernel(Y, Y, gamma)
    K_xy = _rbf_kernel(X, Y, gamma)

    nx = len(X)
    ny = len(Y)

    # Unbiased MMD^2 estimate
    mmd_sq = (
        (K_xx.sum() - np.trace(K_xx)) / (nx * (nx - 1))
        + (K_yy.sum() - np.trace(K_yy)) / (ny * (ny - 1))
        - 2.0 * K_xy.mean()
    )

    mmd = float(np.sqrt(max(mmd_sq, 0.0)))
    logger.debug("MMD (gamma=%.6f): %.6f", gamma, mmd)
    return mmd


def _pairwise_sq_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute matrix of squared Euclidean distances between rows of X and Y."""
    X_norm = (X ** 2).sum(axis=1, keepdims=True)
    Y_norm = (Y ** 2).sum(axis=1, keepdims=True)
    return X_norm + Y_norm.T - 2 * (X @ Y.T)


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """Compute the RBF (Gaussian) kernel matrix K[i,j] = exp(-gamma * ||x_i - y_j||^2)."""
    sq_dists = _pairwise_sq_distances(X, Y)
    return np.exp(-gamma * np.maximum(sq_dists, 0.0))
