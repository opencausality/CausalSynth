"""Structural equation fitting and evaluation.

Each variable X_i in the SCM has the structural equation:
    X_i = intercept + sum(coeff_j * X_j  for j in parents(X_i)) + noise_i

The coefficients and intercept are estimated via OLS regression.
The residuals from that regression parameterise the noise distribution.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from causalsynth.data.schema import StructuralEquation
from causalsynth.exceptions import SCMFitError
from causalsynth.scm.noise import fit_noise_params, sample_noise

logger = logging.getLogger("causalsynth.scm.equations")


def fit_equation(
    variable: str,
    parents: list[str],
    df: pd.DataFrame,
    noise_type: str,
) -> StructuralEquation:
    """Fit a structural equation for one variable using OLS regression on parents.

    If the variable has no parents it is treated as exogenous: the equation
    is just 'X_i = mean(X_i) + noise' where noise is fitted to the
    demeaned values.

    Args:
        variable: Target variable name.
        parents: List of parent variable names.
        df: DataFrame containing the real data.
        noise_type: Noise distribution to fit to residuals.

    Returns:
        A fitted StructuralEquation.

    Raises:
        SCMFitError: If regression fails or data is malformed.
    """
    if variable not in df.columns:
        raise SCMFitError(variable, f"column '{variable}' not found in DataFrame")

    y = df[variable].values.astype(float)

    if len(parents) == 0:
        # Exogenous variable — no parents
        intercept = float(np.mean(y))
        coefficients: dict[str, float] = {}
        residuals = y - intercept

        noise_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1.0
        noise_params = fit_noise_params(residuals, noise_type)

        logger.debug(
            "Fitted exogenous equation for '%s': intercept=%.4f, noise_%s std≈%.4f",
            variable,
            intercept,
            noise_type,
            noise_std,
        )

        eq = StructuralEquation(
            variable=variable,
            parents=[],
            coefficients={},
            intercept=intercept,
            noise_type=noise_type,
            noise_std=noise_params.get("std", noise_std),
            noise_range=(
                noise_params.get("low", -noise_std),
                noise_params.get("high", noise_std),
            ),
        )
        # Attach raw params for sampler
        eq.__pydantic_extra__ = {"_noise_params": noise_params}  # type: ignore[assignment]
        object.__setattr__(eq, "_noise_params", noise_params)
        return eq

    # Endogenous variable — regress on parents
    missing = [p for p in parents if p not in df.columns]
    if missing:
        raise SCMFitError(
            variable,
            f"parent columns missing from DataFrame: {missing}",
        )

    X = df[parents].values.astype(float)

    try:
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
    except Exception as exc:
        raise SCMFitError(variable, f"OLS regression failed: {exc}") from exc

    intercept = float(model.intercept_)
    coefficients = {p: float(c) for p, c in zip(parents, model.coef_)}
    residuals = y - model.predict(X)

    noise_params = fit_noise_params(residuals, noise_type)
    noise_std = noise_params.get("std", float(np.std(residuals, ddof=1)))

    logger.debug(
        "Fitted equation for '%s': intercept=%.4f, coefficients=%s, "
        "noise_%s std≈%.4f",
        variable,
        intercept,
        {k: f"{v:.4f}" for k, v in coefficients.items()},
        noise_type,
        noise_std,
    )

    eq = StructuralEquation(
        variable=variable,
        parents=parents,
        coefficients=coefficients,
        intercept=intercept,
        noise_type=noise_type,
        noise_std=noise_std,
        noise_range=(
            noise_params.get("low", -noise_std),
            noise_params.get("high", noise_std),
        ),
    )
    object.__setattr__(eq, "_noise_params", noise_params)
    return eq


def evaluate_equation(
    eq: StructuralEquation,
    parent_values: dict[str, float],
    rng: np.random.Generator,
) -> float:
    """Evaluate a structural equation for a single sample.

    Computes: X_i = intercept + sum(coeff_j * parent_j) + noise

    Args:
        eq: The structural equation to evaluate.
        parent_values: Mapping of parent_name -> value for this sample.
        rng: NumPy Generator for sampling noise.

    Returns:
        A single float value for X_i.
    """
    value = eq.intercept

    for parent, coef in eq.coefficients.items():
        if parent not in parent_values:
            raise KeyError(
                f"Parent '{parent}' not found in parent_values dict. "
                f"Available: {list(parent_values.keys())}"
            )
        value += coef * parent_values[parent]

    # Get noise params — prefer attached params, fall back to eq fields
    noise_params: dict = getattr(eq, "_noise_params", {})
    if not noise_params:
        if eq.noise_type == "gaussian":
            noise_params = {"std": eq.noise_std}
        elif eq.noise_type == "laplace":
            noise_params = {"scale": eq.noise_std}
        elif eq.noise_type == "uniform":
            noise_params = {"low": eq.noise_range[0], "high": eq.noise_range[1]}

    noise = sample_noise(eq.noise_type, noise_params, n=1, rng=rng)[0]
    return value + noise


def evaluate_equation_batch(
    eq: StructuralEquation,
    parent_values: dict[str, np.ndarray],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evaluate a structural equation for a batch of n samples.

    Args:
        eq: The structural equation to evaluate.
        parent_values: Mapping of parent_name -> array of n values.
        n: Number of samples.
        rng: NumPy Generator.

    Returns:
        Array of n floats.
    """
    values = np.full(n, eq.intercept, dtype=float)

    for parent, coef in eq.coefficients.items():
        if parent not in parent_values:
            raise KeyError(f"Parent '{parent}' not found in batch parent_values.")
        values += coef * parent_values[parent]

    noise_params: dict = getattr(eq, "_noise_params", {})
    if not noise_params:
        if eq.noise_type == "gaussian":
            noise_params = {"std": eq.noise_std}
        elif eq.noise_type == "laplace":
            noise_params = {"scale": eq.noise_std}
        elif eq.noise_type == "uniform":
            noise_params = {"low": eq.noise_range[0], "high": eq.noise_range[1]}

    noise = sample_noise(eq.noise_type, noise_params, n=n, rng=rng)
    return values + noise
