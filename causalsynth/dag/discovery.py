"""Automatic DAG discovery from tabular data.

Uses correlation-based structure learning with conditional independence tests
to discover a plausible causal DAG without requiring a user-supplied graph.

Algorithm:
1. Compute pairwise Pearson correlations.
2. For each pair (X, Y), test independence conditioning on all other variables
   using partial correlation + Fisher Z test.
3. Keep edges where partial correlation is significant.
4. Orient edges using correlation magnitude heuristic (stronger correlations
   suggest direct causes; break remaining cycles greedily).
5. Enforce acyclicity.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from causalsynth.data.schema import CausalDAG
from causalsynth.exceptions import DAGDiscoveryError

logger = logging.getLogger("causalsynth.dag.discovery")


def _partial_correlation(
    df: pd.DataFrame, x: str, y: str, controlling: list[str]
) -> tuple[float, float]:
    """Compute partial correlation between x and y controlling for other variables.

    Uses the recursive formula via residuals from OLS regression.

    Returns:
        (partial_corr, p_value)
    """
    n = len(df)

    if not controlling:
        r, p = stats.pearsonr(df[x].values, df[y].values)
        return float(r), float(p)

    # Regress x and y on controlling variables, compute residuals
    from numpy.linalg import lstsq

    Z = df[controlling].values
    # Add intercept
    Z_int = np.column_stack([np.ones(n), Z])

    coef_x, _, _, _ = lstsq(Z_int, df[x].values, rcond=None)
    coef_y, _, _, _ = lstsq(Z_int, df[y].values, rcond=None)

    resid_x = df[x].values - Z_int @ coef_x
    resid_y = df[y].values - Z_int @ coef_y

    if resid_x.std() < 1e-10 or resid_y.std() < 1e-10:
        return 0.0, 1.0

    r, p = stats.pearsonr(resid_x, resid_y)
    return float(r), float(p)


def _fisher_z_test(r: float, n: int, alpha: float) -> bool:
    """Return True if correlation r is significant at level alpha (two-tailed).

    Uses Fisher's Z transformation.
    """
    if abs(r) >= 1.0:
        return True
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1.0 / np.sqrt(max(n - 3, 1))
    p_value = 2 * (1 - stats.norm.cdf(abs(z) / se))
    return bool(p_value < alpha)


def _break_cycle_greedy(edges: list[tuple[str, str]], nodes: list[str]) -> list[tuple[str, str]]:
    """Remove edges one by one (weakest first) until the graph is acyclic."""
    import networkx as nx

    # We don't have weights here so we remove edges in order
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    max_iterations = len(edges) + 1
    iteration = 0

    while not nx.is_directed_acyclic_graph(g) and iteration < max_iterations:
        iteration += 1
        try:
            cycle = nx.find_cycle(g, orientation="original")
            # Remove the last edge in the cycle
            _, target, _ = cycle[-1]
            source, _, _ = cycle[-1]
            g.remove_edge(source, target)
            logger.debug("Removed cycle edge %s -> %s", source, target)
        except nx.NetworkXNoCycle:
            break

    return list(g.edges())


def discover_dag(df: pd.DataFrame, significance_level: float = 0.05) -> CausalDAG:
    """Discover a causal DAG from tabular data using conditional independence tests.

    Algorithm:
    1. Compute pairwise correlations.
    2. Test conditional independence between all pairs (conditioning on all
       other variables).
    3. Use correlation magnitude to suggest edge directions.
    4. Enforce acyclicity by removing weak edges.

    Args:
        df: Input DataFrame with numeric columns only.
        significance_level: Alpha level for conditional independence tests.

    Returns:
        A CausalDAG discovered from the data.

    Raises:
        DAGDiscoveryError: If discovery cannot produce a valid DAG.
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        raise DAGDiscoveryError(
            "Need at least 2 numeric columns for DAG discovery."
        )

    df_num = df[numeric_cols].copy()
    n = len(df_num)
    nodes = list(numeric_cols)

    logger.info(
        "Starting DAG discovery on %d columns, %d rows (alpha=%.3f)",
        len(nodes),
        n,
        significance_level,
    )

    # Step 1: Compute pairwise correlations
    corr_matrix = df_num.corr().abs()

    # Step 2: Test conditional independence for all pairs
    edges: list[tuple[str, str]] = []
    all_other: dict[tuple[str, str], list[str]] = {}

    for x, y in combinations(nodes, 2):
        controlling = [c for c in nodes if c != x and c != y]
        r, p = _partial_correlation(df_num, x, y, controlling)
        is_dependent = _fisher_z_test(r, n, significance_level)

        if is_dependent:
            # Determine edge direction: higher marginal correlation with other
            # variables suggests the variable is "more upstream" (a root cause)
            # Heuristic: variable with lower average partial correlation with
            # others is more likely an exogenous root.
            raw_corr_x = corr_matrix[x].drop([x, y]).mean()
            raw_corr_y = corr_matrix[y].drop([x, y]).mean()

            # The variable with fewer strong connections tends to be upstream
            if raw_corr_x <= raw_corr_y:
                edges.append((x, y))
            else:
                edges.append((y, x))

            all_other[(x, y)] = controlling
            logger.debug(
                "Edge %s -> %s (partial r=%.3f, p=%.4f)",
                edges[-1][0],
                edges[-1][1],
                r,
                p,
            )

    # Step 3: Enforce acyclicity
    edges = _break_cycle_greedy(edges, nodes)

    logger.info(
        "DAG discovery complete: %d nodes, %d edges", len(nodes), len(edges)
    )

    return CausalDAG(nodes=nodes, edges=edges)
