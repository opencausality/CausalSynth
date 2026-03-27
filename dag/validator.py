"""DAG validation utilities.

Provides assertion-style functions that raise ValidationError on failure,
suitable for use in pipelines where a bad DAG should halt execution.
"""

from __future__ import annotations

import logging

import networkx as nx
import pandas as pd

from causalsynth.data.schema import CausalDAG
from causalsynth.exceptions import ValidationError

logger = logging.getLogger("causalsynth.dag.validator")


def assert_acyclic(dag: CausalDAG) -> None:
    """Assert that a CausalDAG contains no directed cycles.

    Args:
        dag: The DAG to check.

    Raises:
        ValidationError: If the graph contains one or more directed cycles.
    """
    g = nx.DiGraph()
    g.add_nodes_from(dag.nodes)
    g.add_edges_from(dag.edges)

    if not nx.is_directed_acyclic_graph(g):
        try:
            cycles = list(nx.simple_cycles(g))
        except Exception:
            cycles = []

        cycle_descriptions = [" -> ".join(c + [c[0]]) for c in cycles[:3]]
        summary = "; ".join(cycle_descriptions) if cycle_descriptions else "unknown cycles"
        raise ValidationError(
            f"DAG is not acyclic. Found cycles: {summary}"
        )

    logger.debug("DAG acyclicity check passed (%d nodes, %d edges).", len(dag.nodes), len(dag.edges))


def assert_all_nodes_in_data(dag: CausalDAG, df: pd.DataFrame) -> None:
    """Assert that every node in the DAG is a column in the DataFrame.

    Args:
        dag: The causal DAG whose nodes must be present in the data.
        df: The DataFrame to check against.

    Raises:
        ValidationError: If one or more DAG nodes are missing from the DataFrame.
    """
    missing = [node for node in dag.nodes if node not in df.columns]
    if missing:
        raise ValidationError(
            f"The following DAG nodes are missing from the DataFrame: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    logger.debug(
        "All %d DAG nodes found in DataFrame columns.", len(dag.nodes)
    )


def assert_nodes_numeric(dag: CausalDAG, df: pd.DataFrame) -> None:
    """Assert that all DAG nodes correspond to numeric DataFrame columns.

    Warns (rather than raises) for non-numeric columns so callers can decide
    how to handle mixed-type data.

    Args:
        dag: The causal DAG.
        df: The DataFrame.
    """
    non_numeric = [
        node
        for node in dag.nodes
        if node in df.columns and not pd.api.types.is_numeric_dtype(df[node])
    ]
    if non_numeric:
        logger.warning(
            "The following DAG nodes are non-numeric and will be skipped "
            "during SCM fitting: %s",
            non_numeric,
        )
