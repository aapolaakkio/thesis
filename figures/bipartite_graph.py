"""Plot a simple customer-driver bipartite graph."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

FIG_DIR = Path(__file__).resolve().parent

SOURCE = "S"
SINK = "T"
CUSTOMERS = ["C1", "C2", "C3", "C4", "C5"]
DRIVERS = ["D1", "D2", "D3"]
EDGES = [
    ("C1", "D1"),
    ("C1", "D2"),
    ("C2", "D1"),
    ("C3", "D2"),
    ("C3", "D3"),
    ("C4", "D2"),
    ("C5", "D2"),
    ("C5", "D3"),
]
SOURCE_EDGES = [(SOURCE, c) for c in CUSTOMERS]
SINK_EDGES = [(d, SINK) for d in DRIVERS]

CUSTOMER_COLOR = "#dbe9f9"
CUSTOMER_EDGE = "#3a76b5"
DRIVER_COLOR = "#fdeede"
DRIVER_EDGE = "#cb7a2d"
TERMINAL_COLOR = "#e3e0d8"
TERMINAL_EDGE = "#6b6257"


def main() -> None:
    G = nx.Graph()
    G.add_nodes_from(CUSTOMERS, bipartite=0)
    G.add_nodes_from(DRIVERS, bipartite=1)
    G.add_edges_from(EDGES)
    G.add_edges_from(SOURCE_EDGES + SINK_EDGES)

    pos = {c: (0.0, -i) for i, c in enumerate(CUSTOMERS)}
    pos |= {d: (1.0, -i - 1) for i, d in enumerate(DRIVERS)}
    pos[SOURCE] = (-1.0, -(len(CUSTOMERS) - 1) / 2)
    pos[SINK] = (2.0, -(len(DRIVERS) - 1) / 2 - 1)

    fig, ax = plt.subplots(figsize=(5.5, 7))
    nx.draw_networkx_edges(
        G, pos, ax=ax, edgelist=EDGES, edge_color="#666666", width=1.4
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax, edgelist=SOURCE_EDGES + SINK_EDGES,
        edge_color="#999999", width=1.2, style="dashed",
    )
    for nodes, fill, edge in [
        (CUSTOMERS, CUSTOMER_COLOR, CUSTOMER_EDGE),
        (DRIVERS, DRIVER_COLOR, DRIVER_EDGE),
        ([SOURCE, SINK], TERMINAL_COLOR, TERMINAL_EDGE),
    ]:
        nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=nodes, node_size=2000,
            node_color=fill, edgecolors=edge, linewidths=2.2,
        )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=16, font_weight="bold")

    ax.legend(
        handles=[
            Line2D(
                [], [], linestyle="none", marker="o", markersize=15,
                markerfacecolor=CUSTOMER_COLOR, markeredgecolor=CUSTOMER_EDGE,
                markeredgewidth=1.8, label="Customers",
            ),
            Line2D(
                [], [], linestyle="none", marker="o", markersize=15,
                markerfacecolor=DRIVER_COLOR, markeredgecolor=DRIVER_EDGE,
                markeredgewidth=1.8, label="Drivers",
            ),
            Line2D(
                [], [], linestyle="none", marker="o", markersize=15,
                markerfacecolor=TERMINAL_COLOR, markeredgecolor=TERMINAL_EDGE,
                markeredgewidth=1.8, label="Source / sink",
            ),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False,
        fontsize=15,
    )
    ax.set_axis_off()
    ax.margins(x=0.12, y=0.08)

    fig.tight_layout()
    out = FIG_DIR / "bipartite_graph.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
