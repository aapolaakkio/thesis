"""Process-flow figure of the pipeline, sized for a 16:9 slide."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path(__file__).resolve().parent

BLUE_FILL, BLUE_EDGE = "#dbe9f9", "#3a76b5"
GREEN_FILL, GREEN_EDGE = "#dff0e2", "#3f8a4e"
ORANGE_FILL, ORANGE_EDGE = "#fdeede", "#cb7a2d"
PURPLE_FILL, PURPLE_EDGE = "#ece1f5", "#7a4fa3"
GREY = "#444444"
CUT_RED = "#c0392b"

BOX_W, BOX_H = 2.8, 1.5
Y_MID = 4.1
Y_TOP = 8.0
Y_METHOD = 1.5
X_INPUT = 1.7
X_MATCH, X_SCHED, X_CONF = 5.6, 9.7, 13.8
X_OUT = 16.9

DOT_MS = 13
MINI_LW = 1.8


def stage_box(ax, cx, title, fill, edge):
    ax.add_patch(
        FancyBboxPatch(
            (cx - BOX_W / 2, Y_MID - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.1",
            facecolor=fill, edgecolor=edge, linewidth=2.2,
        )
    )
    ax.text(cx, Y_MID, title, ha="center", va="center",
            fontsize=20, fontweight="bold")


def arrow(ax, p0, p1, rad=0.0, color=GREY, lw=2.0, style="-|>", scale=20):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=scale,
            linewidth=lw, color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def dotted_link(ax, cx, y0, y1):
    ax.plot([cx, cx], [y0, y1], linestyle=":", color="#999999", lw=1.4)


L_OFF = [0.85, 0.0, -0.85]
R_OFF = [0.5, -0.55]
MINI_EDGES = [(0, 0), (0, 1), (1, 0), (2, 1)]
MINI_DX = 1.0


def mini_dots(ax, cx, cy):
    lx, rx = cx - MINI_DX, cx + MINI_DX
    for off in L_OFF:
        ax.plot(lx, cy + off, "o", ms=DOT_MS,
                mfc=BLUE_FILL, mec=BLUE_EDGE, mew=2.0)
    for off in R_OFF:
        ax.plot(rx, cy + off, "o", ms=DOT_MS,
                mfc=ORANGE_FILL, mec=ORANGE_EDGE, mew=2.0)


def mini_preference(ax, cx, cy, cut_edges=()):
    """Plain bipartite preference graph; edges in ``cut_edges`` are drawn cut."""
    lx, rx = cx - MINI_DX, cx + MINI_DX
    for e in MINI_EDGES:
        i, j = e
        y0, y1 = cy + L_OFF[i], cy + R_OFF[j]
        if e in cut_edges:
            ax.plot([lx, rx], [y0, y1], color=CUT_RED, lw=MINI_LW,
                    linestyle=(0, (3, 2)))
            mx, my = (lx + rx) / 2, (y0 + y1) / 2
            ax.text(mx, my, r"$\times$", ha="center", va="center",
                    fontsize=18, color=CUT_RED, fontweight="bold")
        else:
            ax.plot([lx, rx], [y0, y1], color="#666666", lw=MINI_LW)
    mini_dots(ax, cx, cy)


def mini_flow(ax, cx, cy):
    """Directed graph with source and sink attached."""
    lx, rx = cx - MINI_DX, cx + MINI_DX
    sx, tx = cx - 1.95, cx + 1.95
    for i, j in MINI_EDGES:
        arrow(ax, (lx, cy + L_OFF[i]), (rx, cy + R_OFF[j]),
              color="#666666", lw=1.4, scale=13)
    for off in L_OFF:
        arrow(ax, (sx, cy), (lx, cy + off), color="#666666", lw=1.4, scale=13)
    for off in R_OFF:
        arrow(ax, (rx, cy + off), (tx, cy), color="#666666", lw=1.4, scale=13)
    mini_dots(ax, cx, cy)
    ax.plot(sx, cy, "o", ms=DOT_MS + 1, mfc=GREEN_FILL, mec=GREEN_EDGE, mew=2.2)
    ax.plot(tx, cy, "o", ms=DOT_MS + 1, mfc=PURPLE_FILL, mec=PURPLE_EDGE, mew=2.2)
    ax.text(sx, cy + 0.55, "$s$", ha="center", fontsize=16)
    ax.text(tx, cy + 0.55, "$t$", ha="center", fontsize=16)


def mini_schedule(ax, cx, cy):
    """Rides as |--| intervals in staggered rows above a time axis."""
    x0, w, tick = cx - 1.7, 3.4, 0.14
    rows = [
        (cy + 0.8, [(0.1, 0.75), (1.2, 0.75), (2.25, 0.85)]),
        (cy + 0.1, [(0.4, 0.95), (1.8, 1.0)]),
    ]
    for y, intervals in rows:
        for bx, bw in intervals:
            xa, xb = x0 + bx, x0 + bx + bw
            ax.plot([xa, xb], [y, y], color=GREEN_EDGE, lw=2.2)
            for x in (xa, xb):
                ax.plot([x, x], [y - tick, y + tick], color=GREEN_EDGE, lw=2.2)
    y_axis = cy - 0.75
    ax.plot([x0, x0 + w], [y_axis, y_axis], color="#555555", lw=1.8)
    for x in (x0, x0 + w):
        ax.plot([x, x], [y_axis - tick, y_axis + tick], color="#555555", lw=1.8)


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 18.4)
    ax.set_ylim(0, 10.35)
    ax.set_aspect("equal")
    ax.set_axis_off()

    ax.text(X_INPUT, Y_MID, "Preference\ngraph", ha="center", va="center",
            fontsize=18, style="italic")
    stage_box(ax, X_MATCH, "Matching", BLUE_FILL, BLUE_EDGE)
    stage_box(ax, X_SCHED, "Scheduling", GREEN_FILL, GREEN_EDGE)
    stage_box(ax, X_CONF, "Conflict\nresolution", ORANGE_FILL, ORANGE_EDGE)
    ax.text(X_OUT, Y_MID, "Final\nschedule", ha="center", va="center",
            fontsize=18, style="italic")

    half = BOX_W / 2 + 0.18
    arrow(ax, (X_INPUT + 1.05, Y_MID), (X_MATCH - half, Y_MID))
    arrow(ax, (X_MATCH + half, Y_MID), (X_SCHED - half, Y_MID))
    arrow(ax, (X_SCHED + half, Y_MID + 0.38), (X_CONF - half, Y_MID + 0.38),
          rad=-0.35)
    arrow(ax, (X_CONF - half, Y_MID - 0.38), (X_SCHED + half, Y_MID - 0.38),
          rad=-0.35)
    ax.text((X_SCHED + X_CONF) / 2, Y_MID - 1.15, "re-solve",
            ha="center", fontsize=16, color=GREY)
    arrow(ax, (X_CONF + half, Y_MID), (X_OUT - 1.0, Y_MID))

    for cx, label in [
        (X_MATCH, "Ford–Fulkerson\n(Edmonds–Karp)"),
        (X_SCHED, "MILP\n(Gurobi solver)"),
    ]:
        dotted_link(ax, cx, Y_MID - BOX_H / 2 - 0.12, Y_METHOD + 0.55)
        ax.text(cx, Y_METHOD, label, ha="center", va="center",
                fontsize=17, color=GREY)

    for cx, draw in [
        (X_INPUT, lambda c: mini_preference(ax, c, Y_TOP)),
        (X_MATCH, lambda c: mini_flow(ax, c, Y_TOP)),
        (X_SCHED, lambda c: mini_schedule(ax, c, Y_TOP)),
        (X_CONF, lambda c: mini_preference(ax, c, Y_TOP,
                                           cut_edges=((0, 1),))),
    ]:
        draw(cx)
        dotted_link(ax, cx, Y_TOP - 1.25,
                    Y_MID + (0.65 if cx == X_INPUT else BOX_H / 2 + 0.12))

    fig.tight_layout()
    out = FIG_DIR / "pipeline_process_flow.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
