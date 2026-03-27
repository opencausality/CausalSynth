"""Differential privacy via the Laplace mechanism.

Adds calibrated Laplace noise to numeric columns so the synthetic data
satisfies epsilon-differential privacy with respect to individual record
sensitivity.

Background:
  The Laplace mechanism adds noise drawn from Laplace(0, sensitivity/epsilon)
  to each numeric feature.  Smaller epsilon = stronger privacy guarantee but
  more distortion.  Typical epsilon values range from 0.1 (strong) to 10 (weak).

Reference:
  Dwork & Roth, "The Algorithmic Foundations of Differential Privacy", 2014.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("causalsynth.generation.privacy")


def add_differential_privacy(
    synthetic: pd.DataFrame,
    epsilon: float,
    sensitivity: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Add Laplace noise for epsilon-differential privacy.

    Applies the Laplace mechanism independently to each numeric column.
    Non-numeric columns are left unchanged.

    Noise scale per column = sensitivity / epsilon.

    Args:
        synthetic: Synthetic DataFrame to protect.
        epsilon: Privacy budget. Smaller = stronger privacy. Must be > 0.
        sensitivity: Global sensitivity of each column (default 1.0).
                     For unbounded data, consider normalising columns first.
        seed: Optional random seed for reproducibility.

    Returns:
        A new DataFrame with Laplace noise added to numeric columns.

    Raises:
        ValueError: If epsilon <= 0.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0 for differential privacy, got {epsilon}")

    scale = sensitivity / epsilon
    rng = np.random.default_rng(seed)

    result = synthetic.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        logger.warning("No numeric columns found; differential privacy noise not added.")
        return result

    for col in numeric_cols:
        noise = rng.laplace(loc=0.0, scale=scale, size=len(result))
        result[col] = result[col] + noise

    logger.info(
        "Added Laplace DP noise to %d columns (epsilon=%.4f, scale=%.6f).",
        len(numeric_cols),
        epsilon,
        scale,
    )
    return result


def privacy_budget_info(epsilon: float, sensitivity: float = 1.0) -> dict:
    """Return a summary of the privacy budget configuration.

    Args:
        epsilon: Privacy parameter.
        sensitivity: Global sensitivity.

    Returns:
        Dictionary with privacy budget information.
    """
    scale = sensitivity / epsilon

    if epsilon < 0.5:
        level = "Strong privacy (high distortion)"
    elif epsilon < 2.0:
        level = "Moderate privacy"
    elif epsilon < 5.0:
        level = "Weak privacy (low distortion)"
    else:
        level = "Minimal privacy (very low distortion)"

    return {
        "epsilon": epsilon,
        "sensitivity": sensitivity,
        "laplace_scale": scale,
        "privacy_level": level,
    }
