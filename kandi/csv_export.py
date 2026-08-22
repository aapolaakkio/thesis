"""CSV export for pipeline results."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from kandi.scheduling import STRATEGIES

METRICS_DIR = Path("metrics")


def timestamped_metrics_csv_path(prefix: str) -> Path:
    """Path for a pipeline metrics CSV: ``metrics/<prefix>-YYYYMMDD-HHMMSS.csv``."""
    return METRICS_DIR / f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}.csv"


_SCHED_INPUTS = (
    "seed",
    "E_max",
    "P_penalty",
    "B_min",
    "B_max",
    "I_idle",
    "D_delay",
    "S_min", "S_max", "S_mean",
    "H_min", "H_max", "H_mean",
    "U_min", "U_max", "U_mean",
    "R_min", "R_max", "R_mean",
)
_SCHED_FINAL = ("total_delay", "n_unassigned", "total_idle")
_SCHED_PASSTHROUGH = _SCHED_INPUTS + _SCHED_FINAL

PIPELINE_CSV_FIELDS = [
    "graph_index",
    "graph_id",
    "matching_n_customers",
    "matching_n_drivers",
    "matching_max_flow",
    "matching_ford_fulkerson_seconds",
    "matching_coverage_pct",
    "matching_avg_customers_per_driver",
    "matching_avg_drivers_per_customer",
    *(f"scheduling_{k}" for k in _SCHED_INPUTS),
    "scheduling_objective",
    "conflict_objective",
    "conflict_chosen_strategy",
    *(f"conflict_obj_{s}" for s in STRATEGIES),
    *(f"conflict_resolve_seconds_{s}" for s in STRATEGIES),
    "scheduling_total_delay",
    "scheduling_n_assigned",
    "scheduling_n_unassigned",
    "scheduling_total_idle",
    "scheduling_drivers_used",
    "scheduling_max_driver_load",
    "scheduling_avg_load_per_used_driver",
    "scheduling_seconds",
    "conflict_seconds",
    "total_seconds",
]

_SCHED_CSV_KEYS = tuple(
    f for f in PIPELINE_CSV_FIELDS if not f.startswith(("graph_", "matching_"))
)


def _final_schedule_stats(sch: dict[str, Any]) -> dict[str, Any]:
    """Derive final-state aggregates from the resolved ``assigned`` map."""
    assigned = sch.get("assigned") or {}
    loads = Counter(k for k in assigned.values() if k is not None)
    drivers_used = len(loads)
    n_assigned = sum(loads.values())
    return {
        "scheduling_n_assigned": n_assigned,
        "scheduling_drivers_used": drivers_used,
        "scheduling_max_driver_load": max(loads.values(), default=0),
        "scheduling_avg_load_per_used_driver": (
            n_assigned / drivers_used if drivers_used else 0.0
        ),
    }


def pipeline_result_to_csv_row(result: dict[str, Any]) -> dict[str, Any]:
    m = result["metrics"]
    row: dict[str, Any] = {
        "graph_index": result["graph_index"],
        "graph_id": result["graph_id"],
        "matching_n_customers": m["n_customers"],
        "matching_n_drivers": m["n_drivers"],
        "matching_max_flow": result["max_flow"],
        "matching_ford_fulkerson_seconds": m["ford_fulkerson_seconds"],
        "matching_coverage_pct": m["coverage_pct"],
        "matching_avg_customers_per_driver": m["avg_customers_per_driver"],
        "matching_avg_drivers_per_customer": m["avg_drivers_per_customer"],
    }
    sch = result.get("scheduling")
    if not sch:
        row.update(dict.fromkeys(_SCHED_CSV_KEYS, ""))
        return row

    strat_objs = sch.get("strategy_objectives") or {}
    strat_walls = sch.get("strategy_resolve_wall_seconds") or {}
    for k in _SCHED_PASSTHROUGH:
        row[f"scheduling_{k}"] = sch.get(k, "")
    for s in STRATEGIES:
        row[f"conflict_obj_{s}"] = strat_objs.get(s, "")
        row[f"conflict_resolve_seconds_{s}"] = strat_walls.get(s, "")
    row["scheduling_objective"] = sch.get("objective_before_resolve", "")
    row["conflict_objective"] = sch.get("objective", "")
    row["conflict_chosen_strategy"] = sch.get("chosen_strategy", "")
    row["scheduling_seconds"] = sch.get("scheduling_wall_seconds", "")
    row["conflict_seconds"] = sch.get("conflict_wall_seconds", "")
    row["total_seconds"] = sch.get("total_wall_seconds", "")
    row.update(_final_schedule_stats(sch))
    return row


def write_pipeline_results_csv(
    results: list[dict[str, Any]], path: Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [pipeline_result_to_csv_row(r) for r in results]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PIPELINE_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path
