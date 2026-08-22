"""Per-graph, batch and generated-graph pipeline entry points."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from kandi.context import build_graph_context, draw_road_graph, print_graph_context
from kandi.graph_io import (
    load_bipartite_graphs_from_file,
    write_complex_bipartite_graph_file,
)
from kandi.csv_export import timestamped_metrics_csv_path, write_pipeline_results_csv
from kandi.max_flow import (
    compute_max_flow,
    extract_customer_driver_flow_matching,
    postprocess_matching_metrics,
)
from kandi.scheduling import attach_job_scheduling_to_pipeline_result

GENERATED_GRAPHS_DIR = Path("graphs")


def process_graph_at_index(
    graphs: list[dict[str, Any]],
    graph_index: int,
    *,
    plot: bool = False,
    verbose: bool = True,
    figsize: tuple[float, float] = (14, 9),
    scheduling_seed: int | None = None,
    scheduling_E_max: float = 15.0,
    scheduling_P_penalty: float = 300.0,
    scheduling_B_min: float = 10.0,
    scheduling_B_max: float = 60.0,
    scheduling_I_idle: float = 1.0,
    scheduling_D_delay: float = 1.0,
    scheduling_S_k_range: tuple[float, float] = (0.0, 60.0),
    scheduling_H_k_range: tuple[float, float] = (240.0, 600.0),
    scheduling_U_range: tuple[float, float] = (0.0, 600.0),
    scheduling_ride_duration: tuple[float, float] = (5.0, 30.0),
    scheduling_time_limit: float | None = 30.0,
    scheduling_mip_gap: float | None = 0.01,
    scheduling_mip_gap_abs: float | None = None,
    scheduling_max_resolve_rounds: int = 5,
    scheduling_time_granularity: float = 1.0,
) -> dict[str, Any]:
    """Run max-flow matching, then the per-driver scheduling MILP."""
    entry = graphs[graph_index]
    if verbose:
        print(f"[pipeline] >>> processing graph index={graph_index} id={entry['id']}")
    ctx = build_graph_context(entry)
    if verbose:
        print_graph_context(ctx, graph_index)
        print("[pipeline] running Ford-Fulkerson max flow...")

    max_flow, G_dir, G_residual, ford_ff_sec = compute_max_flow(
        ctx["G_road"], ctx["SOURCE"], ctx["SINK"]
    )
    if verbose:
        print(
            f"[pipeline]   max flow = {max_flow:g} (FF {ford_ff_sec * 1000:.2f} ms)"
        )
        print("[pipeline] extracting customer<->driver matching...")

    matching = extract_customer_driver_flow_matching(
        ctx["G_road"], G_dir, G_residual, ctx["part"], ctx["road_nodes"]
    )
    metrics = postprocess_matching_metrics(matching, ford_ff_sec)
    if verbose:
        print(
            f"[pipeline]   matching: coverage={metrics['coverage_pct']:.1f}% "
            f"({metrics['n_served_customers']}/{metrics['n_customers']} served), "
            f"avg drivers/customer={metrics['avg_drivers_per_customer']:.2f}, "
            f"avg customers/driver={metrics['avg_customers_per_driver']:.2f}"
        )

    out: dict[str, Any] = {
        "graph_index": graph_index,
        "graph_id": entry["id"],
        "max_flow": max_flow,
        "metrics": metrics,
        "ctx": ctx,
        "G_dir": G_dir,
        "G_residual": G_residual,
        "matching": matching,
    }
    if plot:
        if verbose:
            print("[pipeline] drawing road graph...")
        draw_road_graph(ctx, figsize=figsize, show=True)
    out = attach_job_scheduling_to_pipeline_result(
        out,
        seed=scheduling_seed,
        E_max=scheduling_E_max,
        P_penalty=scheduling_P_penalty,
        B_min=scheduling_B_min,
        B_max=scheduling_B_max,
        I_idle=scheduling_I_idle,
        D_delay=scheduling_D_delay,
        S_k_range=scheduling_S_k_range,
        H_k_range=scheduling_H_k_range,
        U_range=scheduling_U_range,
        ride_duration=scheduling_ride_duration,
        time_limit=scheduling_time_limit,
        mip_gap=scheduling_mip_gap,
        mip_gap_abs=scheduling_mip_gap_abs,
        max_resolve_rounds=scheduling_max_resolve_rounds,
        time_granularity=scheduling_time_granularity,
        progress=verbose,
    )
    if verbose:
        sch = out["scheduling"]
        print(
            f"[pipeline] <<< done: obj={sch['objective']:.2f} "
            f"total_delay={sch['total_delay']:.2f} "
            f"unassigned={sch['n_unassigned']}/{sch['N_jobs']} "
            f"resolved={sch['n_drivers_resolved']} "
            f"wall={sch['total_wall_seconds'] * 1000:.1f}ms"
        )
    return out


def process_all_graphs(
    graphs: list[dict[str, Any]],
    *,
    plot_each: bool = False,
    verbose_each: bool = True,
    csv_path: Path | None = None,
    scheduling_seed: int | None = None,
    scheduling_E_max: float = 15.0,
    scheduling_P_penalty: float = 300.0,
    scheduling_B_min: float = 10.0,
    scheduling_B_max: float = 60.0,
    scheduling_I_idle: float = 1.0,
    scheduling_D_delay: float = 1.0,
    scheduling_S_k_range: tuple[float, float] = (0.0, 60.0),
    scheduling_H_k_range: tuple[float, float] = (240.0, 600.0),
    scheduling_U_range: tuple[float, float] = (0.0, 600.0),
    scheduling_ride_duration: tuple[float, float] = (5.0, 30.0),
    scheduling_time_limit: float | None = 30.0,
    scheduling_mip_gap: float | None = 0.01,
    scheduling_mip_gap_abs: float | None = None,
    scheduling_max_resolve_rounds: int = 5,
    scheduling_time_granularity: float = 1.0,
) -> list[dict[str, Any]]:
    """Run process_graph_at_index on every graph; write a metrics CSV."""
    n = len(graphs)
    if verbose_each:
        print(f"[pipeline] processing {n} graphs...")
    results: list[dict[str, Any]] = []
    for i in range(n):
        if verbose_each:
            print(f"[pipeline] === graph {i + 1}/{n} ===")
        results.append(
            process_graph_at_index(
                graphs,
                i,
                plot=plot_each,
                verbose=verbose_each,
                scheduling_seed=scheduling_seed,
                scheduling_E_max=scheduling_E_max,
                scheduling_P_penalty=scheduling_P_penalty,
                scheduling_B_min=scheduling_B_min,
                scheduling_B_max=scheduling_B_max,
                scheduling_I_idle=scheduling_I_idle,
                scheduling_D_delay=scheduling_D_delay,
                scheduling_S_k_range=scheduling_S_k_range,
                scheduling_H_k_range=scheduling_H_k_range,
                scheduling_U_range=scheduling_U_range,
                scheduling_ride_duration=scheduling_ride_duration,
                scheduling_time_limit=scheduling_time_limit,
                scheduling_mip_gap=scheduling_mip_gap,
                scheduling_mip_gap_abs=scheduling_mip_gap_abs,
                scheduling_max_resolve_rounds=scheduling_max_resolve_rounds,
                scheduling_time_granularity=scheduling_time_granularity,
            )
        )
    if csv_path is None:
        csv_path = timestamped_metrics_csv_path("pipeline_metrics")
    write_pipeline_results_csv(results, csv_path)
    print(f"Wrote metrics CSV: {Path(csv_path).resolve()}")
    return results


def run_complex_test(
    *,
    graph_id: int = 100001,
    n_customers: int = 40,
    n_drivers: int = 20,
    edges_per_customer: tuple[int, int] = (3, 6),
    graph_seed: int = 42,
    out_path: Path | None = None,
    customer_capacity: float = 1.0,
    driver_capacity: float | None = None,
    scheduling_seed: int | None = None,
    scheduling_E_max: float = 15.0,
    scheduling_P_penalty: float = 300.0,
    scheduling_B_min: float = 10.0,
    scheduling_B_max: float = 60.0,
    scheduling_I_idle: float = 1.0,
    scheduling_D_delay: float = 1.0,
    scheduling_S_k_range: tuple[float, float] = (0.0, 60.0),
    scheduling_H_k_range: tuple[float, float] = (240.0, 600.0),
    scheduling_U_range: tuple[float, float] = (0.0, 600.0),
    scheduling_ride_duration: tuple[float, float] = (5.0, 30.0),
    scheduling_time_limit: float | None = 30.0,
    scheduling_mip_gap: float | None = 0.01,
    scheduling_max_resolve_rounds: int = 5,
    scheduling_time_granularity: float = 1.0,
    plot: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate a complex bipartite graph and run the full pipeline on it."""
    if out_path is None:
        out_path = GENERATED_GRAPHS_DIR / f"complex_{uuid4().hex}.graph"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_complex_bipartite_graph_file(
        out_path,
        configs=[
            {
                "graph_id": graph_id,
                "n_customers": n_customers,
                "n_drivers": n_drivers,
                "edges_per_customer_min": edges_per_customer[0],
                "edges_per_customer_max": edges_per_customer[1],
                "seed": graph_seed,
            }
        ],
    )
    new_graphs = load_bipartite_graphs_from_file(
        out_path,
        customer_capacity=customer_capacity,
        driver_capacity=driver_capacity,
    )
    if not new_graphs:
        raise RuntimeError("Generated graph was rejected as non-bipartite.")

    res = process_graph_at_index(
        new_graphs,
        0,
        plot=plot,
        verbose=verbose,
        scheduling_seed=scheduling_seed,
        scheduling_E_max=scheduling_E_max,
        scheduling_P_penalty=scheduling_P_penalty,
        scheduling_B_min=scheduling_B_min,
        scheduling_B_max=scheduling_B_max,
        scheduling_I_idle=scheduling_I_idle,
        scheduling_D_delay=scheduling_D_delay,
        scheduling_S_k_range=scheduling_S_k_range,
        scheduling_H_k_range=scheduling_H_k_range,
        scheduling_U_range=scheduling_U_range,
        scheduling_ride_duration=scheduling_ride_duration,
        scheduling_time_limit=scheduling_time_limit,
        scheduling_mip_gap=scheduling_mip_gap,
        scheduling_max_resolve_rounds=scheduling_max_resolve_rounds,
        scheduling_time_granularity=scheduling_time_granularity,
    )
    if verbose:
        print_complex_test_summary(res)
    return res


def run_complex_test_sweep(
    *,
    customer_range: tuple[int, int, int] | None = None,
    driver_range: tuple[int, int, int] | None = None,
    pairs: list[tuple[int, int]] | None = None,
    base_graph_id: int = 1,
    base_graph_seed: int = 1,
    csv_path: Path | None = None,
    verbose: bool = True,
    **run_kwargs: Any,
) -> list[dict[str, Any]]:
    """Run ``run_complex_test`` over a sweep of (n_customers, n_drivers) configs."""
    forbidden = {"graph_id", "graph_seed", "n_customers", "n_drivers", "verbose"}
    overlap = forbidden & run_kwargs.keys()
    if overlap:
        raise ValueError(
            f"run_kwargs cannot contain {sorted(overlap)} (managed by sweep)"
        )

    if pairs is None and customer_range is None and driver_range is None:
        raise ValueError("Provide pairs, customer_range + driver_range, or both.")
    if (customer_range is None) != (driver_range is None):
        raise ValueError(
            "customer_range and driver_range must be provided together."
        )

    configs: list[tuple[int, int]] = []
    if pairs is not None:
        configs.extend((int(c), int(d)) for c, d in pairs)
    if customer_range is not None and driver_range is not None:
        c_lo, c_hi, c_step = customer_range
        d_lo, d_hi, d_step = driver_range
        if c_step <= 0 or d_step <= 0:
            raise ValueError("Range steps must be positive.")
        if c_hi < c_lo or d_hi < d_lo:
            raise ValueError("Range max must be >= min.")
        c_vals = list(range(c_lo, c_hi + 1, c_step))
        d_vals = list(range(d_lo, d_hi + 1, d_step))
        configs.extend((c, d) for c in c_vals for d in d_vals)

    if not configs:
        raise ValueError("Sweep produced no configs.")

    if verbose:
        print(f"[sweep] running {len(configs)} (n_customers, n_drivers) configs")

    results: list[dict[str, Any]] = []
    for i, (n_c, n_d) in enumerate(configs):
        if verbose:
            print(
                f"[sweep] === config {i + 1}/{len(configs)}: "
                f"customers={n_c}, drivers={n_d} ==="
            )
        res = run_complex_test(
            graph_id=base_graph_id + i,
            graph_seed=base_graph_seed + i,
            n_customers=n_c,
            n_drivers=n_d,
            verbose=verbose,
            **run_kwargs,
        )
        results.append(res)

    if csv_path is None:
        csv_path = timestamped_metrics_csv_path("complex_sweep_metrics")
    write_pipeline_results_csv(results, csv_path)
    if verbose:
        print(f"[sweep] wrote metrics CSV: {Path(csv_path).resolve()}")
    return results


def print_complex_test_summary(res: dict[str, Any]) -> None:
    m = res["metrics"]
    sch = res["scheduling"]
    per_driver: dict[int, list[int]] = defaultdict(list)
    for cust_idx, drv_idx in sch.get("assigned", {}).items():
        if drv_idx is not None:
            per_driver[drv_idx].append(cust_idx)
    loads = [len(v) for v in per_driver.values()]
    multi = sum(1 for v in per_driver.values() if len(v) > 1)
    print(f"\n--- Graph index {res['graph_index']} (id {res['graph_id']}) ---")
    print(
        f"  FF matching: N={m['n_customers']} K={m['n_drivers']} "
        f"coverage={m['coverage_pct']:.1f}% "
        f"drivers/customer={m['avg_drivers_per_customer']:.2f} "
        f"customers/driver={m['avg_customers_per_driver']:.2f}"
    )
    R_lo, R_hi = sch.get("ride_duration_range", (0.0, 0.0))
    print(
        f"  Per-driver MILP: obj={sch['objective']:.2f} "
        f"total_delay={sch['total_delay']:.2f} "
        f"unassigned={sch['n_unassigned']}/{sch['N_jobs']} "
        f"R~U[{R_lo:g},{R_hi:g}] (mean={sch.get('R_mean', 0.0):.2f}) "
        f"wall={sch['scheduling_wall_seconds'] * 1000:.1f} ms"
    )
    print(
        f"  Conflicts:  customers chosen by >1 driver={sch.get('n_conflicts', 0)}, "
        f"losing (driver,customer) pairs dropped={sch.get('n_dropped_pairs', 0)}, "
        f"re-solved drivers={sch.get('n_drivers_resolved', 0)} "
        f"({sch.get('conflict_wall_seconds', 0.0) * 1000:.1f} ms, "
        f"{sch.get('n_resolve_rounds', 0)} rounds, "
        f"delay {sch.get('total_delay_before_resolve', 0.0):.2f} -> {sch.get('total_delay', 0.0):.2f}, "
        f"newly_assigned={sch.get('n_newly_assigned', 0)})"
    )
    print(
        f"  Routes:     drivers used={len(per_driver)}/{m['n_drivers']}, "
        f"with >1 customer={multi}, max load={max(loads, default=0)}"
    )
    print_driver_schedules(res)


def print_driver_schedules(res: dict[str, Any]) -> None:
    """Print each driver's schedule, customers sorted by pickup time."""
    sch = res["scheduling"]
    K = sch["K"]
    U = sch["U"]
    R = sch["R"]
    S = sch["S"]
    H = sch["H"]
    e = sch.get("e", {})
    assigned = sch.get("assigned", {})

    per_driver: dict[int, list[int]] = defaultdict(list)
    for cust_idx, drv_idx in assigned.items():
        if drv_idx is not None:
            per_driver[drv_idx].append(cust_idx)

    print("\n  Driver schedules:")
    for k in range(K):
        custs = per_driver.get(k, [])
        if not custs:
            print(f"    driver {k:>3}  shift=[{S[k]:6.2f}, {H[k]:6.2f}]  (idle)")
            continue
        custs_sorted = sorted(custs, key=lambda i: U[i] + e.get(i, 0.0))
        print(
            f"    driver {k:>3}  shift=[{S[k]:6.2f}, {H[k]:6.2f}]  "
            f"({len(custs_sorted)} customer{'s' if len(custs_sorted) != 1 else ''})"
        )
        for i in custs_sorted:
            ei = e.get(i, 0.0)
            pickup = U[i] + ei
            end = pickup + R[i]
            print(
                f"      cust {i:>3}  U={U[i]:6.2f}  delay={ei:5.2f}  "
                f"pickup={pickup:6.2f}  R={R[i]:5.2f}  end={end:6.2f}"
            )

    unassigned = sorted(i for i, k in assigned.items() if k is None)
    if unassigned:
        print(f"    unassigned ({len(unassigned)}): {unassigned}")
