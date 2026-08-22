"""Plot a road network with waiting customers, drivers and one pickup route."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Wedge

FIG_DIR = Path(__file__).resolve().parent

LOCATIONS = {
    "a": (0.0, 3.0), "b": (1.3, 3.8), "c": (2.8, 3.3),
    "d": (4.2, 4.0), "e": (5.6, 3.4), "f": (7.0, 3.9),
    "g": (0.5, 1.9), "h": (2.0, 2.3), "i": (3.4, 1.8),
    "j": (4.9, 2.4), "k": (6.3, 2.0), "l": (7.6, 2.6),
    "m": (0.2, 0.6), "n": (1.6, 0.9), "o": (3.0, 0.4),
    "p": (4.4, 1.0), "q": (5.8, 0.6), "r": (7.2, 1.1),
    "s": (2.2, -0.7), "t": (3.8, -1.0), "u": (5.3, -0.6),
    "v": (6.8, -0.2),
}
ROADS = [
    ("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"),
    ("g", "h"), ("h", "i"), ("i", "j"), ("j", "k"), ("k", "l"),
    ("m", "n"), ("n", "o"), ("o", "p"), ("p", "q"), ("q", "r"),
    ("s", "t"), ("t", "u"), ("u", "v"),
    ("a", "g"), ("b", "h"), ("c", "i"), ("d", "j"), ("e", "k"), ("f", "l"),
    ("g", "m"), ("h", "n"), ("i", "o"), ("j", "p"), ("k", "q"), ("l", "r"),
    ("n", "s"), ("o", "t"), ("p", "u"), ("q", "v"), ("r", "v"),
    ("b", "g"), ("d", "i"), ("k", "r"),
]
CUSTOMERS = ["d", "i", "m", "q", "v"]
DRIVERS = ["b", "n", "u"]
ROUTE = ["n", "s", "t", "o", "p", "j", "d"]

CUSTOMER_COLOR = "#dbe9f9"
CUSTOMER_EDGE = "#3a76b5"
DRIVER_COLOR = "#fdeede"
DRIVER_EDGE = "#cb7a2d"
ROAD_COLOR = "#b9b9b9"
JUNCTION_COLOR = "#8f8f8f"
ROUTE_COLOR = "#444444"

NODE_R = 0.38
JUNCTION_R = 0.09


def person_patches(cx: float, cy: float, r: float, color: str) -> list:
    """Head-and-shoulders silhouette centred on the node."""
    s = r * 1.05
    return [
        Circle((cx, cy + 0.36 * s), 0.30 * s, facecolor=color, edgecolor="none"),
        Wedge((cx, cy - 0.52 * s), 0.62 * s, 0, 180,
              facecolor=color, edgecolor="none"),
    ]


def car_patches(cx: float, cy: float, r: float, color: str) -> list:
    """Side view of a car centred on the node."""
    s = r * 1.05
    cabin = [(-0.34, 0.10), (-0.20, 0.46), (0.20, 0.46), (0.36, 0.10)]
    return [
        FancyBboxPatch(
            (cx - 0.62 * s, cy - 0.16 * s), 1.24 * s, 0.30 * s,
            boxstyle="round,pad=0,rounding_size=" + str(0.10 * s),
            facecolor=color, edgecolor="none",
        ),
        Polygon([(cx + x * s, cy + y * s) for x, y in cabin],
                closed=True, facecolor=color, edgecolor="none"),
        Circle((cx - 0.34 * s, cy - 0.26 * s), 0.15 * s,
               facecolor=color, edgecolor="none"),
        Circle((cx + 0.34 * s, cy - 0.26 * s), 0.15 * s,
               facecolor=color, edgecolor="none"),
    ]


def draw_occupied(ax, xy: tuple[float, float], kind: str) -> None:
    """A location holding a customer or a driver, drawn as an icon disc."""
    cx, cy = xy
    fill, edge = ((CUSTOMER_COLOR, CUSTOMER_EDGE) if kind == "customer"
                  else (DRIVER_COLOR, DRIVER_EDGE))
    ax.add_patch(Circle((cx, cy), NODE_R, facecolor=fill, edgecolor=edge,
                        linewidth=2.2, zorder=3))
    icons = (person_patches if kind == "customer" else car_patches)
    for patch in icons(cx, cy, NODE_R, edge):
        patch.set_zorder(4)
        ax.add_patch(patch)


def main() -> None:
    route_edges = {frozenset(pair) for pair in zip(ROUTE, ROUTE[1:])}

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    for u, v in ROADS:
        (x0, y0), (x1, y1) = LOCATIONS[u], LOCATIONS[v]
        on_route = frozenset((u, v)) in route_edges
        ax.plot([x0, x1], [y0, y1], zorder=2 if on_route else 1,
                color=ROUTE_COLOR if on_route else ROAD_COLOR,
                linewidth=3.0 if on_route else 1.8,
                solid_capstyle="round")

    occupied = {**{n: "customer" for n in CUSTOMERS},
                **{n: "driver" for n in DRIVERS}}
    for name, xy in LOCATIONS.items():
        if name in occupied:
            draw_occupied(ax, xy, occupied[name])
        else:
            ax.add_patch(Circle(xy, JUNCTION_R, facecolor=JUNCTION_COLOR,
                                edgecolor="none", zorder=3))

    ax.legend(
        handles=[
            Line2D([], [], linestyle="none", marker="o", markersize=14,
                   markerfacecolor=CUSTOMER_COLOR,
                   markeredgecolor=CUSTOMER_EDGE, markeredgewidth=1.8,
                   label="Waiting customer"),
            Line2D([], [], linestyle="none", marker="o", markersize=14,
                   markerfacecolor=DRIVER_COLOR, markeredgecolor=DRIVER_EDGE,
                   markeredgewidth=1.8, label="Available driver"),
            Line2D([], [], linestyle="none", marker="o", markersize=6,
                   markerfacecolor=JUNCTION_COLOR, markeredgecolor="none",
                   label="Location"),
            Line2D([], [], color=ROAD_COLOR, linewidth=1.8, label="Road"),
            Line2D([], [], color=ROUTE_COLOR, linewidth=3.0,
                   label="Pickup route"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False,
        fontsize=12,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.autoscale_view()
    ax.margins(x=0.06, y=0.10)

    fig.tight_layout()
    out = FIG_DIR / "ridesharing_illustration.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
