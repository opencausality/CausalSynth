"""DAG loading utilities.

Supports loading causal DAGs from JSON files with the schema:
    {"nodes": ["A", "B", "C"], "edges": [["A", "B"], ["B", "C"]]}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx

from causalsynth.data.schema import CausalDAG
from causalsynth.exceptions import DAGLoadError, ValidationError

logger = logging.getLogger("causalsynth.dag.loader")


def load_dag(path: Path) -> CausalDAG:
    """Load a CausalDAG from a JSON file.

    The JSON file must have the format:
        {
            "nodes": ["age", "bmi", "blood_pressure"],
            "edges": [["age", "bmi"], ["age", "blood_pressure"]]
        }

    Args:
        path: Path to the JSON file.

    Returns:
        A validated CausalDAG instance.

    Raises:
        DAGLoadError: If the file cannot be read or parsed.
        ValidationError: If the DAG contains cycles.
    """
    path = Path(path)

    if not path.exists():
        raise DAGLoadError(str(path), "file does not exist")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DAGLoadError(str(path), f"cannot read file: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DAGLoadError(str(path), f"invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise DAGLoadError(str(path), "top-level JSON must be an object")

    if "nodes" not in data:
        raise DAGLoadError(str(path), "missing 'nodes' key")
    if "edges" not in data:
        raise DAGLoadError(str(path), "missing 'edges' key")

    # Normalise edges to list[tuple[str, str]]
    try:
        edges: list[tuple[str, str]] = [
            (str(e[0]), str(e[1])) for e in data["edges"]
        ]
    except (TypeError, IndexError) as exc:
        raise DAGLoadError(
            str(path), f"edges must be a list of [cause, effect] pairs: {exc}"
        ) from exc

    try:
        dag = CausalDAG(nodes=[str(n) for n in data["nodes"]], edges=edges)
    except Exception as exc:
        raise DAGLoadError(str(path), f"schema validation failed: {exc}") from exc

    logger.info(
        "Loaded DAG from '%s': %d nodes, %d edges",
        path,
        len(dag.nodes),
        len(dag.edges),
    )

    validate_dag(dag)
    return dag


def validate_dag(dag: CausalDAG) -> None:
    """Validate that a CausalDAG is acyclic.

    Args:
        dag: The DAG to validate.

    Raises:
        ValidationError: If the graph contains cycles.
    """
    g = nx.DiGraph()
    g.add_nodes_from(dag.nodes)
    g.add_edges_from(dag.edges)

    if not nx.is_directed_acyclic_graph(g):
        cycles = list(nx.simple_cycles(g))
        cycle_strs = [" -> ".join(c) for c in cycles[:3]]
        raise ValidationError(
            f"DAG contains cycles: {'; '.join(cycle_strs)}"
        )

    logger.debug("DAG passed acyclicity validation.")


def get_topological_order(dag: CausalDAG) -> list[str]:
    """Return variables in topological order (parents before children).

    Args:
        dag: A validated (acyclic) CausalDAG.

    Returns:
        Ordered list of variable names.

    Raises:
        ValidationError: If the graph contains cycles.
    """
    g = nx.DiGraph()
    g.add_nodes_from(dag.nodes)
    g.add_edges_from(dag.edges)

    try:
        order = list(nx.topological_sort(g))
    except nx.NetworkXUnfeasible as exc:
        raise ValidationError(f"Cannot compute topological order: {exc}") from exc

    logger.debug("Topological order: %s", order)
    return order
