"""Ancestral sampling from a fitted SCM.

Generates synthetic data by traversing the DAG in topological order and
sampling each variable from its structural equation given previously-sampled
parent values.

This approach:
- Guarantees that all causal dependencies are exactly preserved
- Is efficient: O(n_samples * n_variables)
- Is deterministic given a fixed seed
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from causalsynth.data.schema import SCM
from causalsynth.scm.equations import evaluate_equation_batch

logger = logging.getLogger("causalsynth.generation.sampler")


def generate_samples(
    scm: SCM,
    n_samples: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic data by ancestral sampling from the SCM.

    Algorithm:
    1. Initialise RNG with the given seed.
    2. For each variable in topological order:
       a. Gather already-sampled parent values.
       b. Evaluate structural equation: X_i = f(parents) + noise.
    3. Return a DataFrame with one column per variable.

    Args:
        scm: A fitted Structural Causal Model.
        n_samples: Number of synthetic rows to generate.
        seed: Random seed for full reproducibility.

    Returns:
        DataFrame with n_samples rows and one column per SCM variable.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    rng = np.random.default_rng(seed)

    logger.info(
        "Generating %d synthetic samples (seed=%d, %d variables)",
        n_samples,
        seed,
        len(scm.topological_order),
    )

    # Map variable name -> its StructuralEquation for quick lookup
    eq_map = {eq.variable: eq for eq in scm.equations}

    sampled: dict[str, np.ndarray] = {}

    for variable in scm.topological_order:
        if variable not in eq_map:
            logger.warning("No equation found for '%s'; skipping.", variable)
            continue

        eq = eq_map[variable]

        # Gather parent arrays (already sampled in earlier iterations)
        parent_arrays: dict[str, np.ndarray] = {}
        for parent in eq.parents:
            if parent not in sampled:
                raise RuntimeError(
                    f"Parent '{parent}' of '{variable}' has not been sampled yet. "
                    "This indicates a topological order violation."
                )
            parent_arrays[parent] = sampled[parent]

        values = evaluate_equation_batch(eq, parent_arrays, n_samples, rng)
        sampled[variable] = values

        logger.debug(
            "Sampled '%s': mean=%.4f, std=%.4f",
            variable,
            float(np.mean(values)),
            float(np.std(values)),
        )

    # Build DataFrame in topological order
    df = pd.DataFrame({var: sampled[var] for var in scm.topological_order if var in sampled})

    logger.info(
        "Sampling complete: shape=%s, columns=%s",
        df.shape,
        list(df.columns),
    )
    return df
