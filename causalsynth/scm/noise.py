"""Noise distribution fitting and sampling for structural equations.

Supports three distributions:
- gaussian: Normal(0, std)
- laplace:  Laplace(0, scale)  — heavier tails than Gaussian
- uniform:  Uniform(lo, hi)    — bounded, symmetric around 0
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger("causalsynth.scm.noise")


def fit_noise_params(residuals: np.ndarray, noise_type: str) -> dict:
    """Fit noise distribution parameters to an array of residuals.

    Args:
        residuals: 1-D array of regression residuals.
        noise_type: One of "gaussian", "laplace", "uniform".

    Returns:
        Dictionary of distribution parameters.
        - gaussian: {"std": float}
        - laplace:  {"scale": float}
        - uniform:  {"low": float, "high": float}

    Notes:
        The mean/location is always fixed at 0 because residuals should be
        zero-mean by construction (OLS regression includes an intercept).
    """
    if len(residuals) == 0:
        logger.warning("Empty residuals array; returning default noise params.")
        return _default_params(noise_type)

    residuals = np.asarray(residuals, dtype=float)

    if noise_type == "gaussian":
        std = float(np.std(residuals, ddof=1))
        # Guard against degenerate case
        std = max(std, 1e-6)
        params = {"std": std}

    elif noise_type == "laplace":
        # MAD-based scale estimate: scale = MAD / ln(2)
        mad = float(np.median(np.abs(residuals - np.median(residuals))))
        scale = mad / np.log(2) if mad > 0 else float(np.std(residuals, ddof=1))
        scale = max(scale, 1e-6)
        params = {"scale": scale}

    elif noise_type == "uniform":
        # Fit symmetric uniform based on 5th–95th percentile range
        low = float(np.percentile(residuals, 5))
        high = float(np.percentile(residuals, 95))
        # Ensure it's symmetric and non-degenerate
        half = max(abs(low), abs(high), 1e-6)
        params = {"low": -half, "high": half}

    else:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. "
            "Expected 'gaussian', 'laplace', or 'uniform'."
        )

    logger.debug("Fitted %s noise params: %s", noise_type, params)
    return params


def sample_noise(
    noise_type: str, params: dict, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample n noise values from the specified fitted distribution.

    Args:
        noise_type: One of "gaussian", "laplace", "uniform".
        params: Parameters returned by fit_noise_params.
        n: Number of samples to draw.
        rng: NumPy random Generator for reproducibility.

    Returns:
        1-D float array of length n.
    """
    if noise_type == "gaussian":
        return rng.normal(loc=0.0, scale=params["std"], size=n)

    elif noise_type == "laplace":
        return rng.laplace(loc=0.0, scale=params["scale"], size=n)

    elif noise_type == "uniform":
        return rng.uniform(low=params["low"], high=params["high"], size=n)

    else:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. "
            "Expected 'gaussian', 'laplace', or 'uniform'."
        )


def _default_params(noise_type: str) -> dict:
    """Return safe default parameters when fitting is not possible."""
    if noise_type == "gaussian":
        return {"std": 1.0}
    elif noise_type == "laplace":
        return {"scale": 1.0}
    elif noise_type == "uniform":
        return {"low": -1.0, "high": 1.0}
    return {"std": 1.0}
