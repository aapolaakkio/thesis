"""Generate the results figures from a sweep metrics CSV."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

CSV_DEFAULT = Path("metrics/complex_sweep_metrics-20260707-181307.csv")
FIG_DIR = Path(__file__).resolve().parent

STRATEGIES = [
    "lowest_index",
    "load_asc",
    "load_desc",
    "conflicts_asc",
    "conflicts_desc",
]
STRATEGY_LABELS = {
    "lowest_index": "index",
    "load_asc": "load $\\uparrow$",
    "load_desc": "load $\\downarrow$",
    "conflicts_asc": "confl. $\\uparrow$",
    "conflicts_desc": "confl. $\\downarrow$",
}

REFERENCE = (1000, 200)

PHASE_INITIAL = "#3b6ea5"
PHASE_CONFLICT = "#d1495b"

STRATEGY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
    }
)


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    return float(row[key])


def _i(row: dict, key: str) -> int:
    return int(float(row[key]))


def _grid(rows: list[dict]) -> list[dict]:
    return [r for r in rows if (_i(r, "matching_n_customers"),
                                _i(r, "matching_n_drivers")) != REFERENCE]


def _rows_sorted(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (_i(r, "matching_n_customers"),
                                       _i(r, "matching_n_drivers")))


def _instance_label(r: dict) -> str:
    return f'{_i(r, "matching_n_customers")}/{_i(r, "matching_n_drivers")}'


def plot_runtime_vs_nk(rows: list[dict], out: Path, col: str, title: str) -> None:
    """Two panels: ``col`` (seconds) vs customer count | vs driver count."""
    grid = _grid(rows)
    fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(11, 4.5))

    driver_counts = sorted({_i(r, "matching_n_drivers") for r in grid})
    cmap = plt.get_cmap("viridis")
    for idx, K in enumerate(driver_counts):
        pts = sorted((_i(r, "matching_n_customers"), _f(r, col))
                     for r in grid if _i(r, "matching_n_drivers") == K)
        xs, ys = zip(*pts)
        ax_c.plot(xs, ys, "o-", color=cmap(idx / max(1, len(driver_counts) - 1)),
                  label=f"$K={K}$", markersize=5)
    ax_c.set_xlabel("Customers $N$")
    ax_c.set_ylabel(f"{title} runtime (s)")
    ax_c.set_title("(a) vs. customer count")
    ax_c.legend(title="Drivers", fontsize=8, ncol=2)

    customer_counts = sorted({_i(r, "matching_n_customers") for r in grid})
    cmap2 = plt.get_cmap("plasma")
    for idx, N in enumerate(customer_counts):
        pts = sorted((_i(r, "matching_n_drivers"), _f(r, col))
                     for r in grid if _i(r, "matching_n_customers") == N)
        xs, ys = zip(*pts)
        ax_d.plot(xs, ys, "s-", color=cmap2(idx / max(1, len(customer_counts) - 1)),
                  label=f"$N={N}$", markersize=5)
    ax_d.set_xlabel("Drivers $K$")
    ax_d.set_ylabel(f"{title} runtime (s)")
    ax_d.set_title("(b) vs. driver count")
    ax_d.legend(title="Customers", fontsize=8)

    fig.suptitle(f"{title} runtime")
    fig.tight_layout()
    _save(fig, out)


def plot_resolve_boxplot(rows: list[dict], out: Path) -> None:
    """Boxplot of each strategy's objective, normalised to the per-instance best."""
    excess = {s: [] for s in STRATEGIES}
    for r in rows:
        objs = {s: _f(r, f"conflict_obj_{s}") for s in STRATEGIES}
        best = min(objs.values())
        if best <= 0:
            continue
        for s in STRATEGIES:
            excess[s].append(100.0 * (objs[s] / best - 1.0))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(
        [excess[s] for s in STRATEGIES],
        tick_labels=[STRATEGY_LABELS[s] for s in STRATEGIES],
        showmeans=True,
        showfliers=False,
        patch_artist=True,
        meanprops={"marker": "D", "markerfacecolor": "#1b3a5b",
                   "markeredgecolor": "#1b3a5b", "markersize": 5},
        medianprops={"color": "#1b3a5b", "linewidth": 1.4},
    )
    for i, color in enumerate(STRATEGY_COLORS):
        bp["boxes"][i].set_facecolor(color)
        bp["boxes"][i].set_alpha(0.55)
        bp["boxes"][i].set_edgecolor(color)
        for line in bp["whiskers"][2 * i:2 * i + 2] + bp["caps"][2 * i:2 * i + 2]:
            line.set_color(color)
            line.set_linewidth(1.2)
    ax.set_ylabel("Excess over per-instance best objective (%)")
    ax.set_xlabel("Driver-ordering strategy")
    ax.set_title("Objective by conflict-resolution strategy")
    ax.axhline(0.0, color="grey", lw=0.8, ls="--")
    fig.tight_layout()
    _save(fig, out)


def plot_objective_by_phase(rows: list[dict], out: Path) -> None:
    """Objective before and after conflict resolution, per instance."""
    rows = _rows_sorted(rows)
    labels = [_instance_label(r) for r in rows]
    init = [_f(r, "scheduling_objective") for r in rows]
    final = [_f(r, "conflict_objective") for r in rows]
    improve = [100.0 * (i - f) / i if i else 0.0 for i, f in zip(init, final)]
    x = range(len(rows))

    fig, (ax_abs, ax_rel) = plt.subplots(
        2, 1, figsize=(12, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    width = 0.4
    ax_abs.bar([i - width / 2 for i in x], init, width,
               label="After initial scheduling", color=PHASE_INITIAL)
    ax_abs.bar([i + width / 2 for i in x], final, width,
               label="After conflict resolution", color=PHASE_CONFLICT)
    ax_abs.set_ylabel("Objective (minutes)", fontsize=14)
    ax_abs.set_title("Objective before and after conflict resolution",
                     fontsize=16)
    ax_abs.legend(fontsize=13)

    colors = [PHASE_CONFLICT if v >= 0 else "#8c8c8c" for v in improve]
    ax_rel.bar(list(x), improve, color=colors)
    ax_rel.axhline(0.0, color="grey", lw=0.8)
    ax_rel.set_ylabel("Improvement (%)", fontsize=14)
    ax_rel.set_xlabel("Instance ($N$/$K$)", fontsize=14)
    ax_rel.set_xticks(list(x))
    ax_rel.set_xticklabels(labels, rotation=45, ha="right",
                           rotation_mode="anchor", fontsize=12)
    for ax in (ax_abs, ax_rel):
        ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    _save(fig, out)


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')}")


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_DEFAULT
    rows = load_rows(csv_path)
    print(f"loaded {len(rows)} rows from {csv_path}")

    plot_runtime_vs_nk(rows, FIG_DIR / "results_scheduling_runtime",
                       "scheduling_seconds", "Scheduling")
    plot_runtime_vs_nk(rows, FIG_DIR / "results_resolve_runtime",
                       "conflict_seconds", "Conflict-resolution")
    plot_resolve_boxplot(rows, FIG_DIR / "results_resolve_boxplot")
    plot_objective_by_phase(rows, FIG_DIR / "results_objective_by_phase")


if __name__ == "__main__":
    main()
