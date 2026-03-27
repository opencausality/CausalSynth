"""SCM fitting: calibrate a Structural Causal Model to real data.

For each variable in topological order:
  1. Identify its parents from the DAG.
  2. Regress the variable on its parents (OLS).
  3. Fit a noise distribution to the residuals.
  4. Store coefficients, intercept, and noise params in a StructuralEquation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from causalsynth.dag.loader import get_topological_order
from causalsynth.dag.validator import assert_acyclic, assert_all_nodes_in_data
from causalsynth.data.schema import CausalDAG, SCM
from causalsynth.exceptions import SCMFitError
from causalsynth.scm.equations import fit_equation

logger = logging.getLogger("causalsynth.scm.builder")


def _build_parent_map(dag: CausalDAG) -> dict[str, list[str]]:
    """Return mapping node -> list of parent nodes."""
    parent_map: dict[str, list[str]] = {node: [] for node in dag.nodes}
    for cause, effect in dag.edges:
        parent_map[effect].append(cause)
    return parent_map


def _collect_feature_stats(df: pd.DataFrame, nodes: list[str]) -> dict[str, dict]:
    """Collect per-variable statistics for post-processing."""
    stats: dict[str, dict] = {}
    for node in nodes:
        if node not in df.columns:
            continue
        col = df[node]
        stats[node] = {
            "mean": float(col.mean()),
            "std": float(col.std(ddof=1)) if len(col) > 1 else 1.0,
            "min": float(col.min()),
            "max": float(col.max()),
            "dtype": str(col.dtype),
        }
    return stats


def fit_scm(
    df: pd.DataFrame,
    dag: CausalDAG,
    noise_type: str = "gaussian",
) -> SCM:
    """Fit a Structural Causal Model to real data.

    For each variable in topological order:
    1. Regress variable on its DAG parents using OLS.
    2. Extract regression coefficients and intercept.
    3. Fit a noise distribution to the residuals.

    Args:
        df: Real data DataFrame. All DAG nodes must be columns.
        dag: The causal DAG specifying variable relationships.
        noise_type: Noise distribution to fit. One of "gaussian", "laplace",
                    "uniform".

    Returns:
        A fitted SCM ready for sample generation.

    Raises:
        ValidationError: If the DAG is cyclic or nodes are missing.
        SCMFitError: If regression fails for any variable.
    """
    logger.info(
        "Fitting SCM on %d rows, %d variables, noise='%s'",
        len(df),
        len(dag.nodes),
        noise_type,
    )

    # Validate inputs
    assert_acyclic(dag)
    assert_all_nodes_in_data(dag, df)

    # Work only with numeric DAG columns
    numeric_nodes = [
        n for n in dag.nodes
        if n in df.columns and pd.api.types.is_numeric_dtype(df[n])
    ]
    if not numeric_nodes:
        raise SCMFitError(
            "<all>",
            "No numeric columns found among DAG nodes. "
            "CausalSynth requires numeric variables.",
        )

    if len(numeric_nodes) < len(dag.nodes):
        skipped = set(dag.nodes) - set(numeric_nodes)
        logger.warning("Skipping non-numeric DAG nodes: %s", skipped)

    # Build parent map and topological order
    parent_map = _build_parent_map(dag)
    topo_order = get_topological_order(dag)
    # Filter topo order to numeric nodes only
    topo_order = [n for n in topo_order if n in numeric_nodes]

    equations = []
    for variable in topo_order:
        parents = [p for p in parent_map[variable] if p in numeric_nodes]

        try:
            eq = fit_equation(variable, parents, df, noise_type)
        except SCMFitError:
            raise
        except Exception as exc:
            raise SCMFitError(variable, str(exc)) from exc

        equations.append(eq)
        logger.debug(
            "Fitted equation for '%s' with parents=%s", variable, parents
        )

    feature_stats = _collect_feature_stats(df, dag.nodes)

    scm = SCM(
        dag=dag,
        equations=equations,
        topological_order=topo_order,
        feature_stats=feature_stats,
    )

    logger.info(
        "SCM fitting complete: %d structural equations", len(equations)
    )
    return scm
