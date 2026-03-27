"""Interactive and static DAG visualisation.

Provides two rendering backends:
- pyvis: Interactive HTML (default) — opens in a browser.
- matplotlib: Static PNG — for headless/CI environments.
"""

from __future__ import annotations

import logging
from pathlib import Path

from causalsynth.data.schema import CausalDAG

logger = logging.getLogger("causalsynth.graph.visualizer")

# Node colour palette
NODE_COLOR = "#4a9eff"
ROOT_COLOR = "#ff7043"     # Nodes with no parents (exogenous)
LEAF_COLOR = "#66bb6a"     # Nodes with no children (terminal outcomes)
EDGE_COLOR = "#888888"
BACKGROUND = "#1e1e2e"
FONT_COLOR = "#ffffff"


def render_dag_pyvis(
    dag: CausalDAG,
    output_path: Path = Path("causalsynth_graph.html"),
    title: str = "CausalSynth — Causal DAG",
    height: str = "600px",
    width: str = "100%",
) -> Path:
    """Render the DAG as an interactive HTML file using pyvis.

    Args:
        dag: The CausalDAG to visualise.
        output_path: Where to write the HTML file.
        title: Title shown at the top of the page.
        height: Height of the network canvas.
        width: Width of the network canvas.

    Returns:
        Path to the generated HTML file.
    """
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise ImportError(
            "pyvis is required for interactive visualisation. "
            "Install with: pip install pyvis"
        ) from exc

    # Determine root and leaf nodes
    parents_of = {node: [] for node in dag.nodes}
    children_of = {node: [] for node in dag.nodes}
    for cause, effect in dag.edges:
        parents_of[effect].append(cause)
        children_of[cause].append(effect)

    root_nodes = {n for n in dag.nodes if not parents_of[n]}
    leaf_nodes = {n for n in dag.nodes if not children_of[n]}

    net = Network(
        height=height,
        width=width,
        bgcolor=BACKGROUND,
        font_color=FONT_COLOR,
        directed=True,
        notebook=False,
    )

    # Add nodes
    for node in dag.nodes:
        if node in root_nodes:
            color = ROOT_COLOR
            title_text = f"{node}\n(exogenous root)"
        elif node in leaf_nodes:
            color = LEAF_COLOR
            title_text = f"{node}\n(outcome)"
        else:
            color = NODE_COLOR
            title_text = f"{node}\n(mediator)"

        net.add_node(
            node,
            label=node,
            title=title_text,
            color=color,
            size=25,
            font={"size": 14, "color": FONT_COLOR},
        )

    # Add edges
    for cause, effect in dag.edges:
        net.add_edge(
            cause,
            effect,
            color=EDGE_COLOR,
            arrows="to",
            width=2,
        )

    # Physics layout
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "hierarchicalRepulsion": {
          "centralGravity": 0.0,
          "springLength": 120,
          "springConstant": 0.01,
          "nodeDistance": 150,
          "damping": 0.09
        },
        "solver": "hierarchicalRepulsion"
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "levelSeparation": 120,
          "nodeSpacing": 120,
          "treeSpacing": 200,
          "direction": "UD",
          "sortMethod": "directed"
        }
      }
    }
    """)

    output_path = Path(output_path)
    net.save_graph(str(output_path))

    logger.info("Interactive DAG saved to '%s'.", output_path)
    return output_path


def render_dag_matplotlib(
    dag: CausalDAG,
    output_path: Path = Path("causalsynth_graph.png"),
    figsize: tuple[int, int] = (10, 8),
    dpi: int = 150,
) -> Path:
    """Render the DAG as a static PNG using matplotlib + networkx.

    Args:
        dag: The CausalDAG to visualise.
        output_path: Where to write the PNG file.
        figsize: Figure size in inches (width, height).
        dpi: DPI for the output PNG.

    Returns:
        Path to the generated PNG file.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Headless backend
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "matplotlib and networkx are required for static visualisation."
        ) from exc

    g = nx.DiGraph()
    g.add_nodes_from(dag.nodes)
    g.add_edges_from(dag.edges)

    # Use hierarchical layout
    try:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    except Exception:
        # Fall back to spring layout if graphviz is not installed
        try:
            pos = _hierarchical_layout(g)
        except Exception:
            pos = nx.spring_layout(g, seed=42)

    # Determine node colours
    parents_of: dict[str, list[str]] = {n: [] for n in dag.nodes}
    children_of: dict[str, list[str]] = {n: [] for n in dag.nodes}
    for cause, effect in dag.edges:
        parents_of[effect].append(cause)
        children_of[cause].append(effect)

    node_colors = []
    for n in g.nodes:
        if not parents_of.get(n):
            node_colors.append("#ff7043")   # root
        elif not children_of.get(n):
            node_colors.append("#66bb6a")   # leaf
        else:
            node_colors.append("#4a9eff")   # mediator

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    nx.draw_networkx(
        g,
        pos=pos,
        ax=ax,
        node_color=node_colors,
        edge_color="#aaaaaa",
        font_color="white",
        font_size=11,
        node_size=1800,
        arrows=True,
        arrowsize=20,
        width=2.0,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
    )

    ax.set_title("CausalSynth — Causal DAG", color="white", fontsize=14)
    ax.axis("off")

    output_path = Path(output_path)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.info("Static DAG saved to '%s'.", output_path)
    return output_path


def _hierarchical_layout(g: "nx.DiGraph") -> dict:
    """Compute a simple top-down hierarchical layout without graphviz."""
    import networkx as nx

    layers: dict[str, int] = {}
    for node in nx.topological_sort(g):
        predecessors = list(g.predecessors(node))
        if not predecessors:
            layers[node] = 0
        else:
            layers[node] = max(layers[p] for p in predecessors) + 1

    max_layer = max(layers.values()) if layers else 0
    layer_nodes: dict[int, list[str]] = {}
    for node, layer in layers.items():
        layer_nodes.setdefault(layer, []).append(node)

    pos = {}
    for layer, nodes in layer_nodes.items():
        y = 1.0 - (layer / max(max_layer, 1))
        for i, node in enumerate(nodes):
            x = (i + 1) / (len(nodes) + 1)
            pos[node] = (x, y)

    return pos
