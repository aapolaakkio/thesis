"""Per-graph context (customers vs drivers on road nodes) and matplotlib drawing."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Patch

from kandi.graph_io import road_bipartition

CUSTOMER_PART, DRIVER_PART = 0, 1


def build_graph_context(entry: dict) -> dict[str, Any]:
    """Partition road nodes into customers (color 0) / drivers (color 1)."""
    G_road = entry["graph"]
    SOURCE = entry["source"]
    SINK = entry["sink"]
    part, road_nodes = road_bipartition(G_road, SOURCE, SINK)
    n_customers = sum(1 for n in road_nodes if part[n] == CUSTOMER_PART)
    n_drivers = sum(1 for n in road_nodes if part[n] == DRIVER_PART)
    return {
        "entry": entry,
        "G_road": G_road,
        "SOURCE": SOURCE,
        "SINK": SINK,
        "part": part,
        "road_nodes": road_nodes,
        "n_customers": n_customers,
        "n_drivers": n_drivers,
    }


def print_graph_context(ctx: dict[str, Any], graph_index: int) -> None:
    selected = ctx["entry"]
    G_road = ctx["G_road"]
    print(
        f"Graph id={selected['id']} (index {graph_index}), "
        f"|V|={G_road.number_of_nodes()}, |E|={G_road.number_of_edges()}, "
        f"super-source={ctx['SOURCE']}, super-sink={ctx['SINK']}"
    )
    print(
        f"Mapping: color {CUSTOMER_PART} = customers ({ctx['n_customers']} road nodes), "
        f"color {DRIVER_PART} = drivers ({ctx['n_drivers']} road nodes); "
        f"super-source connects to customers, super-sink to drivers."
    )


def draw_road_graph(
    ctx: dict[str, Any],
    *,
    figsize: tuple[float, float] = (14, 9),
    show: bool = True,
):
    """Two-column bipartite layout; super-source left, super-sink right."""
    G_road = ctx["G_road"]
    SOURCE = ctx["SOURCE"]
    SINK = ctx["SINK"]
    part = ctx["part"]
    road_nodes = ctx["road_nodes"]

    left_nodes = [n for n in road_nodes if part[n] == CUSTOMER_PART]
    G_core = G_road.subgraph(road_nodes).copy()
    pos = nx.bipartite_layout(
        G_core,
        left_nodes,
        align="vertical",
        scale=3.0,
        aspect_ratio=1.0,
    )
    xs = [pos[n][0] for n in road_nodes]
    ys = [pos[n][1] for n in road_nodes]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w = xmax - xmin if xmax > xmin else 1.0
    ymid = 0.5 * (ymin + ymax)
    pos[SOURCE] = np.array([xmin - 0.45 * w, ymid])
    pos[SINK] = np.array([xmax + 0.45 * w, ymid])
    PART_COLORS = ("#5b9bd5", "#f4a582")
    nodes = list(G_road.nodes())
    node_colors = []
    for n in nodes:
        if n == SOURCE:
            node_colors.append(PART_COLORS[0])
        elif n == SINK:
            node_colors.append(PART_COLORS[1])
        else:
            node_colors.append(PART_COLORS[part[n]])

    edgecolors = []
    linewidths = []
    for n in nodes:
        if n == SOURCE or n == SINK:
            edgecolors.append("#1a1a1a")
            linewidths.append(2.8)
        else:
            edgecolors.append("#333333")
            linewidths.append(1.0)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(
        G_road, pos, ax=ax, edge_color="#888888", width=0.35, alpha=0.35, arrows=False
    )
    nx.draw_networkx_nodes(
        G_road,
        pos,
        ax=ax,
        nodelist=nodes,
        node_color=node_colors,
        node_size=280,
        edgecolors=edgecolors,
        linewidths=linewidths,
    )
    nx.draw_networkx_labels(G_road, pos, ax=ax, font_size=7)

    ax.legend(
        handles=[
            Patch(
                facecolor=PART_COLORS[0],
                edgecolor="black",
                linewidth=0.5,
                label="Customers (partition 0)",
            ),
            Patch(
                facecolor=PART_COLORS[1],
                edgecolor="black",
                linewidth=0.5,
                label="Drivers (partition 1)",
            ),
            Patch(
                facecolor=PART_COLORS[0],
                edgecolor="#1a1a1a",
                linewidth=1.2,
                label="Super-source (to customers)",
            ),
            Patch(
                facecolor=PART_COLORS[1],
                edgecolor="#1a1a1a",
                linewidth=1.2,
                label="Super-sink (from drivers)",
            ),
        ],
        title="Customers vs drivers",
        loc="upper left",
        framealpha=0.95,
    )

    edge_labels = {}
    for u, v, d in G_road.edges(data=True):
        if SOURCE in (u, v) or SINK in (u, v):
            continue
        w = d["weight"]
        edge_labels[(u, v)] = f"{w:.0f}" if isinstance(w, float) else str(w)
    nx.draw_networkx_edge_labels(
        G_road, pos, edge_labels, ax=ax, font_size=5, label_pos=0.3, alpha=0.75
    )
    ax.set_axis_off()
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax, pos
