"""Causal structure validation.

Checks whether the synthetic data preserves the causal edges declared in
the DAG by testing whether partial correlations between cause and effect
(controlling for all other variables) remain statistically significant.

An edge X -> Y is considered "preserved" if:
  partial_corr(X, Y | all others) is significant at the given alpha level
  in the SYNTHETIC data.

An edge is "broken" if the synthetic data fails to reproduce this partial
dependency.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from causalsynth.data.schema import CausalDAG

logger = logging.getLogger("causalsynth.validation.causal")


def _partial_correlation_and_pvalue(
    df: pd.DataFrame, x: str, y: str, controlling: list[str]
) -> tuple[float, float]:
    """Compute partial correlation r(x, y | controlling) and its p-value.

    Uses OLS residualisation: regress both x and y on controlling variables,
    then correlate the residuals.

    Returns:
        (partial_corr, p_value) tuple.
    """
    n = len(df)

    if not controlling:
        r, p = stats.pearsonr(df[x].values, df[y].values)
        return float(r), float(p)

    Z = df[controlling].values.astype(float)
    Z_int = np.column_stack([np.ones(n), Z])

    from numpy.linalg import lstsq

    coef_x, _, _, _ = lstsq(Z_int, df[x].values.astype(float), rcond=None)
    coef_y, _, _, _ = lstsq(Z_int, df[y].values.astype(float), rcond=None)

    resid_x = df[x].values.astype(float) - Z_int @ coef_x
    resid_y = df[y].values.astype(float) - Z_int @ coef_y

    std_x = float(np.std(resid_x))
    std_y = float(np.std(resid_y))

    if std_x < 1e-10 or std_y < 1e-10:
        # Degenerate: one residual is constant → not correlated
        return 0.0, 1.0

    r, p = stats.pearsonr(resid_x, resid_y)
    return float(r), float(p)


def validate_causal_structure(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    dag: CausalDAG,
    significance_level: float = 0.05,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Validate whether synthetic data preserves the causal edges in the DAG.

    For each directed edge (X -> Y) in the DAG:
    - Compute partial correlation of X and Y in the SYNTHETIC data,
      conditioning on all other numeric DAG variables.
    - If the partial correlation is statistically significant (p < alpha),
      the edge is considered preserved.
    - Otherwise the edge is considered broken.

    Args:
        real: Original real DataFrame (not used for the test, kept for API
              consistency and future use).
        synthetic: Synthetic DataFrame to validate.
        dag: The ground-truth causal DAG.
        significance_level: Alpha level for significance tests (default 0.05).

    Returns:
        (preserved_edges, broken_edges) where each is a list of (cause, effect)
        tuples.
    """
    # Select only numeric columns present in both synthetic and DAG
    numeric_dag_nodes = [
        n
        for n in dag.nodes
        if n in synthetic.columns
        and pd.api.types.is_numeric_dtype(synthetic[n])
    ]

    if len(numeric_dag_nodes) < 2:
        logger.warning(
            "Fewer than 2 numeric DAG nodes in synthetic data; "
            "cannot validate causal structure."
        )
        return [], list(dag.edges)

    df_synth = synthetic[numeric_dag_nodes].copy()

    preserved: list[tuple[str, str]] = []
    broken: list[tuple[str, str]] = []

    for cause, effect in dag.edges:
        if cause not in numeric_dag_nodes or effect not in numeric_dag_nodes:
            logger.warning(
                "Skipping edge (%s -> %s): not in numeric columns.", cause, effect
            )
            broken.append((cause, effect))
            continue

        # Condition on all other numeric DAG nodes
        controlling = [n for n in numeric_dag_nodes if n != cause and n != effect]
        r, p = _partial_correlation_and_pvalue(df_synth, cause, effect, controlling)

        if p < significance_level:
            preserved.append((cause, effect))
            logger.debug(
                "Edge %s -> %s PRESERVED (partial r=%.4f, p=%.4f)",
                cause,
                effect,
                r,
                p,
            )
        else:
            broken.append((cause, effect))
            logger.debug(
                "Edge %s -> %s BROKEN (partial r=%.4f, p=%.4f)",
                cause,
                effect,
                r,
                p,
            )

    logger.info(
        "Causal validation: %d/%d edges preserved.",
        len(preserved),
        len(dag.edges),
    )
    return preserved, broken
