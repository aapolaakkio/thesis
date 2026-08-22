"""Per-driver time-indexed scheduling MILP (Gurobi). Distance-free; ride-time only.

For each driver k with allowed customer set A_k, time is discretized into
slots of width dt (default 1 minute). x_{i,t} = 1 iff customer i's ride
starts at slot t. T_i = feasible start slots for customer i on driver k:

  T_i = { t : max(U_i, S_k) <= t*dt,  t*dt + R_i <= H_k,  t*dt - U_i <= E_max }

  min  D * sum_{i,t} (t*dt - U_i) * x_{i,t}  +  P * sum_i (1 - y_i)
       +  I * ( (H_k - S_k) - sum_{i,t} R_i * x_{i,t} )
  s.t. y_i = sum_{t in T_i} x_{i,t} <= 1                              forall i
       sum_{(i,t) covering tau (with B_min extension)} x_{i,t} <= 1   forall slot tau
       x_{i,t} <= sum_{(j,t_j): end_i < t_j*dt <= end_i + B_max} x_{j,t_j}
                                          forall (i,t) with end_i + B_max < H_k
       x_{i,t} in {0,1},  y_i in [0,1]

where (i,t) covers slot tau iff t <= tau < t + ceil((R_i + B_min) / dt) and
end_i = t*dt + R_i.

B_min is the minimum break between consecutive rides, enforced by extending
each ride's slot coverage; B_max the maximum, enforced by requiring a
successor unless the ride ends within B_max of H_k. D is the per-minute delay
multiplier, I the per-minute idle penalty on the unused shift.

After the K MILPs, a greedy resolver picks one driver per customer (lowest
delay, tie-break lowest index).
"""

from __future__ import annotations

import bisect
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import gurobipy as gp
import numpy as np
from gurobipy import GRB


_GUROBI_ENV: "gp.Env | None" = None


def _silent_env() -> "gp.Env":
    global _GUROBI_ENV
    if _GUROBI_ENV is None:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        _GUROBI_ENV = env
    return _GUROBI_ENV


def _gurobi_status_name(st: int) -> str:
    names = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(st, str(st))


def driver_to_allowed_customers_from_matching(
    result: dict[str, Any],
) -> dict[int, list[int]]:
    """Map driver index -> allowed customer indices, from the FF matching."""
    m = result["matching"]
    customer_nodes = m["customer_nodes"]
    driver_nodes = m["driver_nodes"]
    d_to_customers = m["driver_to_customers"]
    c_to_i = {c: i for i, c in enumerate(customer_nodes)}
    return {
        k: sorted(c_to_i[c] for c in d_to_customers.get(d, []) if c in c_to_i)
        for k, d in enumerate(driver_nodes)
    }


def generate_scheduling_instance(
    result: dict[str, Any],
    *,
    seed: int | None = None,
    E_max: float = 15.0,
    P_penalty: float = 300.0,
    B_min: float = 10.0,
    B_max: float = 60.0,
    I_idle: float = 1.0,
    D_delay: float = 1.0,
    S_k_range: tuple[float, float] = (0.0, 60.0),
    H_k_range: tuple[float, float] = (240.0, 600.0),
    U_range: tuple[float, float] = (0.0, 600.0),
    ride_duration: tuple[float, float] = (5.0, 30.0),
) -> dict[str, Any]:
    """Sample shifts (S_k, H_k), wanted times U_i and ride durations R_i, in
    minutes. Deterministic in ``seed``, which defaults to the graph id."""
    matching = result["matching"]
    customer_nodes = matching["customer_nodes"]
    driver_nodes = matching["driver_nodes"]
    N = len(customer_nodes)
    K = len(driver_nodes)

    gid = int(result["graph_id"])
    rng = np.random.default_rng(gid if seed is None else int(seed))

    S_lo, S_hi = float(S_k_range[0]), float(S_k_range[1])
    S = rng.uniform(S_lo, S_hi, size=K)

    R_lo, R_hi = float(ride_duration[0]), float(ride_duration[1])
    R = rng.uniform(R_lo, R_hi, size=N) if N else np.zeros(0)

    H_lo, H_hi = float(H_k_range[0]), float(H_k_range[1])
    H = rng.uniform(H_lo, H_hi, size=K)

    U_lo, U_hi = float(U_range[0]), float(U_range[1])
    U = rng.uniform(U_lo, U_hi, size=N) if N else np.zeros(0)

    return {
        "N": N,
        "K": K,
        "U": U,
        "S": S,
        "H": H,
        "R": R,
        "E_max": float(E_max),
        "P": float(P_penalty),
        "B_min": float(B_min),
        "B_max": float(B_max),
        "I_idle": float(I_idle),
        "D_delay": float(D_delay),
        "ride_duration_range": (R_lo, R_hi),
        "S_k_range": (S_lo, S_hi),
        "U_range": (float(U_lo), float(U_hi)),
        "seed": gid if seed is None else int(seed),
        "customer_nodes": customer_nodes,
        "driver_nodes": driver_nodes,
    }


def _bmax_rows(
    customer_slots: dict[int, list[int]],
    R: Any,
    dt: float,
    B_max: float,
    H_k: float,
) -> list[tuple[int, int, list[int], list[tuple[int, int]]]]:
    """Yield ``(i, t_i, full_customers, straddler_pairs)`` per ride needing a
    B_max successor. Requires each customer's slots to be a contiguous range."""
    eps = 1e-9
    bounds = sorted(
        (slots[0] * dt, slots[-1] * dt, j, slots)
        for j, slots in customer_slots.items()
    )
    min_starts = [b[0] for b in bounds]
    max_width = max((b[1] - b[0] for b in bounds), default=0.0)

    rows: list[tuple[int, int, list[int], list[tuple[int, int]]]] = []
    for i, slots_i in customer_slots.items():
        Ri = float(R[i])
        for t_i in slots_i:
            end_i = t_i * dt + Ri
            if end_i + B_max >= H_k - eps:
                continue
            lo = end_i
            hi = end_i + B_max
            a = bisect.bisect_left(min_starts, lo - max_width - eps)
            b = bisect.bisect_right(min_starts, hi + eps)
            full: list[int] = []
            straddlers: list[tuple[int, int]] = []
            for idx in range(a, b):
                ms, ms_max, j, slots_j = bounds[idx]
                if j == i:
                    continue
                if ms_max <= lo - eps or ms > hi + eps:
                    continue
                if ms > lo - eps and ms_max <= hi + eps:
                    full.append(j)
                else:
                    for t_j in slots_j:
                        if lo - eps < t_j * dt <= hi + eps:
                            straddlers.append((j, t_j))
            rows.append((i, t_i, full, straddlers))
    return rows


def solve_single_driver_milp(
    instance: dict[str, Any],
    driver_idx: int,
    customer_indices: list[int],
    *,
    time_limit: float | None = None,
    mip_gap: float | None = None,
    mip_gap_abs: float | None = None,
    verbose: bool = False,
    time_granularity: float = 1.0,
    presolve: int = 1,
    pre_sparsify: int = 0,
    aggregate: int = 0,
) -> dict[str, Any]:
    """Time-indexed MILP for a single driver over their allowed customer subset."""
    U = instance["U"]
    R = instance["R"]
    S_k = float(instance["S"][driver_idx])
    H_k = float(instance["H"][driver_idx])
    E_max = float(instance["E_max"])
    P_pen = float(instance["P"])
    B_min = float(instance.get("B_min", 10.0))
    B_max = float(instance.get("B_max", 60.0))
    I_idle = float(instance.get("I_idle", 1.0))
    D_delay = float(instance.get("D_delay", 1.0))
    dt = float(time_granularity)

    shift_len = max(0.0, H_k - S_k)
    EMPTY = {
        "status": GRB.LOADED,
        "status_name": "EMPTY",
        "objective": I_idle * shift_len,
        "total_delay": 0.0,
        "idle_time": shift_len,
        "n_assigned": 0,
        "assigned_customers": [],
        "e": {},
        "x": {},
        "M": dt,
        "gurobi_seconds": 0.0,
    }

    if not customer_indices:
        return EMPTY

    customer_slots: dict[int, list[int]] = {}
    R_slots: dict[int, int] = {}
    R_slots_with_break: dict[int, int] = {}
    break_slots = max(0, int(math.ceil(B_min / dt - 1e-9)))
    for i in customer_indices:
        Ui = float(U[i])
        Ri = float(R[i])
        t_lo = max(Ui, S_k)
        t_hi = min(Ui + E_max, H_k - Ri)
        if t_lo > t_hi + 1e-9:
            continue
        slot_lo = int(math.ceil(t_lo / dt - 1e-9))
        slot_hi = int(math.floor(t_hi / dt + 1e-9))
        if slot_lo > slot_hi:
            continue
        customer_slots[i] = list(range(slot_lo, slot_hi + 1))
        R_slots[i] = max(1, int(math.ceil(Ri / dt)))
        R_slots_with_break[i] = R_slots[i] + break_slots

    if not customer_slots:
        return EMPTY

    customer_indices = list(customer_slots.keys())

    model = gp.Model(f"sched_driver_{driver_idx}", env=_silent_env())
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.Presolve = presolve
    model.Params.PreSparsify = pre_sparsify
    model.Params.Aggregate = aggregate
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if mip_gap_abs is not None:
        model.Params.MIPGapAbs = float(mip_gap_abs)

    x: dict[tuple[int, int], Any] = {}
    for i in customer_indices:
        for t in customer_slots[i]:
            x[i, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")

    served: dict[int, Any] = {
        i: gp.quicksum(x[i, t] for t in customer_slots[i]) for i in customer_indices
    }
    for i in customer_indices:
        model.addConstr(served[i] <= 1, name=f"assign_{i}")

    y: dict[int, Any] = {}
    for i in customer_indices:
        y[i] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"y_{i}")
        model.addConstr(y[i] == served[i], name=f"served_{i}")

    coverage: dict[int, list[tuple[int, int]]] = {}
    for i in customer_indices:
        ri = R_slots_with_break[i]
        for t in customer_slots[i]:
            for tau in range(t, t + ri):
                coverage.setdefault(tau, []).append((i, t))
    for tau, items in coverage.items():
        if len(items) >= 2:
            model.addConstr(
                gp.quicksum(x[i, t] for (i, t) in items) <= 1,
                name=f"cap_{tau}",
            )

    for i, t_i, full, straddlers in _bmax_rows(customer_slots, R, dt, B_max, H_k):
        model.addConstr(
            x[i, t_i]
            <= gp.quicksum(y[j] for j in full)
            + gp.quicksum(x[j, t_j] for (j, t_j) in straddlers),
            name=f"bmax_{i}_{t_i}",
        )

    delay_term = D_delay * gp.quicksum(
        (t * dt - float(U[i])) * x[i, t]
        for i in customer_indices
        for t in customer_slots[i]
    )
    pen_term = P_pen * gp.quicksum(1 - served[i] for i in customer_indices)
    work_time = gp.quicksum(
        float(R[i]) * x[i, t]
        for i in customer_indices
        for t in customer_slots[i]
    )
    idle_term = I_idle * (shift_len - work_time)
    model.setObjective(delay_term + pen_term + idle_term, GRB.MINIMIZE)

    t0 = time.perf_counter()
    model.optimize()
    gurobi_seconds = time.perf_counter() - t0

    status = model.Status
    out: dict[str, Any] = {
        "status": status,
        "status_name": _gurobi_status_name(status),
        "M": dt,
        "gurobi_seconds": gurobi_seconds,
    }
    has_sol = model.SolCount > 0 and status not in (
        GRB.INFEASIBLE,
        GRB.INF_OR_UNBD,
        GRB.UNBOUNDED,
    )
    if has_sol:
        e_val: dict[int, float] = {}
        x_val: dict[tuple[int, int], float] = {}
        assigned: list[int] = []
        work_total = 0.0
        for i in customer_indices:
            picked_t: int | None = None
            for t in customer_slots[i]:
                xv = float(x[i, t].X)
                x_val[i, t] = xv
                if xv > 0.5:
                    picked_t = t
            if picked_t is not None:
                assigned.append(i)
                e_val[i] = max(0.0, picked_t * dt - float(U[i]))
                work_total += float(R[i])
        out.update(
            {
                "objective": float(model.ObjVal),
                "total_delay": float(sum(e_val.values())),
                "idle_time": max(0.0, shift_len - work_total),
                "n_assigned": len(assigned),
                "assigned_customers": assigned,
                "e": e_val,
                "x": x_val,
            }
        )
    else:
        out.update(
            {
                "objective": float("nan"),
                "total_delay": float("nan"),
                "idle_time": float("nan"),
                "n_assigned": 0,
                "assigned_customers": [],
                "e": {},
                "x": {},
            }
        )

    model.dispose()
    return out


STRATEGIES: tuple[str, ...] = (
    "lowest_index",
    "load_asc",
    "load_desc",
    "conflicts_asc",
    "conflicts_desc",
)


def _driver_order(
    strategy: str,
    assigned: dict[int, int | None],
    K: int,
    conflicts_by_driver: dict[int, int],
) -> list[int]:
    """Driver iteration order for one resolve round, per strategy."""
    if strategy == "lowest_index":
        return list(range(K))
    if strategy in ("load_asc", "load_desc"):
        loads = [sum(1 for kk in assigned.values() if kk == k) for k in range(K)]
        reverse = strategy == "load_desc"
        return sorted(
            range(K),
            key=lambda k: (-loads[k] if reverse else loads[k], k),
        )
    if strategy in ("conflicts_asc", "conflicts_desc"):
        reverse = strategy == "conflicts_desc"
        return sorted(
            range(K),
            key=lambda k: (
                -conflicts_by_driver.get(k, 0) if reverse else conflicts_by_driver.get(k, 0),
                k,
            ),
        )
    raise ValueError(f"Unknown strategy: {strategy!r}")


def _objective_helpers(instance: dict[str, Any], K: int):
    """Build the ``(objective_for, aggregate_idle)`` pair for an instance."""
    P_pen = float(instance["P"])
    I_idle = float(instance.get("I_idle", 1.0))
    D_delay = float(instance.get("D_delay", 1.0))
    H_arr = instance["H"]
    S_arr = instance["S"]
    R_arr = instance["R"]

    def aggregate_idle(assignment: dict[int, int | None]) -> float:
        load_R = [0.0] * K
        for i_idx, k_idx in assignment.items():
            if k_idx is not None:
                load_R[k_idx] += float(R_arr[i_idx])
        return float(
            sum(
                max(0.0, float(H_arr[k]) - float(S_arr[k]) - load_R[k])
                for k in range(K)
            )
        )

    def objective_for(
        assignment: dict[int, int | None], e_map: dict[int, float]
    ) -> float:
        total_delay = float(sum(e_map.values()))
        total_idle = aggregate_idle(assignment)
        n_un = sum(1 for k in assignment.values() if k is None)
        return float(D_delay * total_delay + P_pen * n_un + I_idle * total_idle)

    return objective_for, aggregate_idle


def _run_resolve_strategy(
    strategy: str,
    *,
    cached_milp,
    objective_for,
    aggregate_idle,
    assigned_base: dict[int, int | None],
    e_all_base: dict[int, float],
    per_driver_base: dict[int, dict[str, Any]],
    last_input_set_base: dict[int, frozenset[int]],
    conflicts_by_driver: dict[int, int],
    driver_to_customers: dict[int, list[int]],
    K: int,
    max_resolve_rounds: int,
    progress: bool = False,
) -> dict[str, Any]:
    """Run one driver-ordering strategy's iterated re-solve from the shared
    post-greedy state. ``cached_milp`` maps (k, customers) -> solution."""
    t_resolve = time.perf_counter()
    assigned = dict(assigned_base)
    e_all = dict(e_all_base)
    per_driver = dict(per_driver_base)
    last_input_set = dict(last_input_set_base)

    n_drivers_resolved = 0
    n_drivers_skipped = 0
    n_newly_assigned = 0
    n_rounds_run = 0

    if max_resolve_rounds <= 0:
        return {
            "strategy": strategy,
            "assigned": assigned,
            "e_all": e_all,
            "per_driver": per_driver,
            "n_drivers_resolved": 0,
            "n_drivers_skipped": 0,
            "n_newly_assigned": 0,
            "resolve_wall_seconds": time.perf_counter() - t_resolve,
            "n_rounds_run": 0,
            "objective": objective_for(assigned, e_all),
            "total_delay": float(sum(e_all.values())),
            "total_idle": aggregate_idle(assigned),
        }

    unassigned_pool: set[int] = {i for i, kk in assigned.items() if kk is None}

    if progress:
        print(
            f"[scheduling] [{strategy}] resolving "
            f"(pool={len(unassigned_pool)}, max_rounds={max_resolve_rounds})"
        )

    prev_pool_size = len(unassigned_pool)
    for round_idx in range(max_resolve_rounds):
        round_resolved = 0
        round_newly_assigned = 0
        round_dropped = 0
        round_picks_changed = 0

        driver_iter_order = _driver_order(
            strategy, assigned, K, conflicts_by_driver
        )

        for k in driver_iter_order:
            current_picks = {i for i, kk in assigned.items() if kk == k}
            allowed_k = set(driver_to_customers.get(k, []))
            eligible_new = unassigned_pool & allowed_k
            resolve_input = sorted(current_picks | eligible_new)
            last_picks_set = set(per_driver[k].get("assigned_customers", []))
            new_input_set = frozenset(resolve_input)
            if new_input_set == last_input_set.get(k):
                n_drivers_skipped += 1
                continue
            sol_resolved = cached_milp(k, resolve_input)
            per_driver[k] = sol_resolved
            last_input_set[k] = new_input_set
            round_resolved += 1

            new_picks = set(sol_resolved.get("assigned_customers", []))
            if new_picks != last_picks_set:
                round_picks_changed += 1
            for i in resolve_input:
                if i in new_picks:
                    if assigned.get(i) is None:
                        round_newly_assigned += 1
                        unassigned_pool.discard(i)
                    assigned[i] = k
                    e_all[i] = float(sol_resolved.get("e", {}).get(i, 0.0))
                else:
                    if assigned.get(i) == k:
                        assigned[i] = None
                        e_all.pop(i, None)
                        unassigned_pool.add(i)
                        round_dropped += 1

        n_drivers_resolved += round_resolved
        n_newly_assigned += round_newly_assigned
        n_rounds_run = round_idx + 1

        if progress:
            cur_obj = objective_for(assigned, e_all)
            print(
                f"[scheduling] [{strategy}] round {round_idx + 1}: "
                f"resolved={round_resolved} changed={round_picks_changed} "
                f"newly={round_newly_assigned} dropped={round_dropped} "
                f"pool={len(unassigned_pool)} obj={cur_obj:.2f}"
            )

        if round_picks_changed == 0:
            break
        new_pool_size = len(unassigned_pool)
        if round_idx > 0 and new_pool_size >= prev_pool_size:
            break
        prev_pool_size = new_pool_size

    return {
        "strategy": strategy,
        "assigned": assigned,
        "e_all": e_all,
        "per_driver": per_driver,
        "n_drivers_resolved": n_drivers_resolved,
        "n_drivers_skipped": n_drivers_skipped,
        "n_newly_assigned": n_newly_assigned,
        "resolve_wall_seconds": time.perf_counter() - t_resolve,
        "n_rounds_run": n_rounds_run,
        "objective": objective_for(assigned, e_all),
        "total_delay": float(sum(e_all.values())),
        "total_idle": aggregate_idle(assigned),
    }


_WORKER_INSTANCE: dict[str, Any] | None = None
_WORKER_SOLVE_KWARGS: dict[str, Any] | None = None


def _init_scheduling_worker(
    instance: dict[str, Any], solve_kwargs: dict[str, Any]
) -> None:
    """Pool initializer: stash the instance and solver kwargs per process."""
    global _WORKER_INSTANCE, _WORKER_SOLVE_KWARGS
    _WORKER_INSTANCE = instance
    _WORKER_SOLVE_KWARGS = solve_kwargs


def _worker_context() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the per-process (instance, solve_kwargs)."""
    if _WORKER_INSTANCE is None or _WORKER_SOLVE_KWARGS is None:
        raise RuntimeError(
            "scheduling worker called outside a pool initialized by "
            "_init_scheduling_worker"
        )
    return _WORKER_INSTANCE, _WORKER_SOLVE_KWARGS


def _solve_driver_in_worker(k: int, custs: list[int]) -> tuple[int, dict[str, Any]]:
    """Solve one driver's MILP in a child process; ``(k, solution)`` without
    the per-(i,t) ``x`` map, which is unused downstream."""
    instance, solve_kwargs = _worker_context()
    sol = solve_single_driver_milp(instance, k, custs, **solve_kwargs)
    return k, {kk: vv for kk, vv in sol.items() if kk != "x"}


def _resolve_strategy_in_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one strategy's resolve in a child process."""
    instance, solve_kwargs = _worker_context()
    K = int(instance["K"])
    objective_for, aggregate_idle = _objective_helpers(instance, K)

    cache: dict[tuple[int, frozenset[int]], dict[str, Any]] = {}

    def cached_milp(k: int, custs: list[int]) -> dict[str, Any]:
        key = (k, frozenset(custs))
        hit = cache.get(key)
        if hit is not None:
            replay = dict(hit)
            replay["gurobi_seconds"] = 0.0
            return replay
        sol = solve_single_driver_milp(instance, k, custs, **solve_kwargs)
        cache[key] = sol
        return sol

    result = _run_resolve_strategy(
        payload["strategy"],
        cached_milp=cached_milp,
        objective_for=objective_for,
        aggregate_idle=aggregate_idle,
        assigned_base=payload["assigned_base"],
        e_all_base=payload["e_all_base"],
        per_driver_base=payload["per_driver_base"],
        last_input_set_base=payload["last_input_set_base"],
        conflicts_by_driver=payload["conflicts_by_driver"],
        driver_to_customers=payload["driver_to_customers"],
        K=K,
        max_resolve_rounds=payload["max_resolve_rounds"],
        progress=False,
    )
    result["per_driver"] = {
        k: {kk: vv for kk, vv in sol.items() if kk != "x"}
        for k, sol in result["per_driver"].items()
    }
    return result


def solve_per_driver_scheduling(
    instance: dict[str, Any],
    driver_to_customers: dict[int, list[int]],
    *,
    time_limit: float | None = 30.0,
    mip_gap: float | None = 0.01,
    mip_gap_abs: float | None = None,
    verbose: bool = False,
    max_resolve_rounds: int = 5,
    time_granularity: float = 1.0,
    progress: bool = True,
    presolve: int = 1,
    pre_sparsify: int = 0,
    aggregate: int = 0,
    parallel: bool | None = None,
) -> dict[str, Any]:
    """One MILP per driver, greedy conflict resolution, then five re-solve
    strategies from the same post-greedy state; lowest objective wins."""
    K = instance["K"]
    N = instance["N"]
    _objective_for, _aggregate_idle = _objective_helpers(instance, K)

    use_parallel = (
        parallel
        if parallel is not None
        else (N >= 1000 and (os.cpu_count() or 1) > 1)
    )
    _worker_solve_kwargs = {
        "time_limit": time_limit,
        "mip_gap": mip_gap,
        "mip_gap_abs": mip_gap_abs,
        "verbose": False,
        "time_granularity": time_granularity,
        "presolve": presolve,
        "pre_sparsify": pre_sparsify,
        "aggregate": aggregate,
    }

    milp_cache: dict[tuple[int, frozenset[int]], dict[str, Any]] = {}
    cache_hits = 0
    cache_misses = 0

    def _cached_milp(k: int, custs: list[int]) -> dict[str, Any]:
        nonlocal cache_hits, cache_misses
        key = (k, frozenset(custs))
        hit = milp_cache.get(key)
        if hit is not None:
            cache_hits += 1
            replay = dict(hit)
            replay["gurobi_seconds"] = 0.0
            return replay
        cache_misses += 1
        sol = solve_single_driver_milp(
            instance,
            k,
            custs,
            time_limit=time_limit,
            mip_gap=mip_gap,
            mip_gap_abs=mip_gap_abs,
            verbose=verbose,
            time_granularity=time_granularity,
            presolve=presolve,
            pre_sparsify=pre_sparsify,
            aggregate=aggregate,
        )
        milp_cache[key] = sol
        return sol

    t_first_pass = time.perf_counter()
    per_driver_base: dict[int, dict[str, Any]] = {}
    last_input_set_base: dict[int, frozenset[int]] = {}
    statuses: list[str] = []
    candidates: dict[int, list[tuple[int, float]]] = {i: [] for i in range(N)}
    cust_lists = [list(driver_to_customers.get(k, [])) for k in range(K)]

    first_pass_solutions: dict[int, dict[str, Any]] | None = None
    if use_parallel and K > 1:
        fp_workers = min(K, os.cpu_count() or 1)
        if progress:
            print(
                f"[scheduling] solving {K} per-driver MILPs (N={N}) in "
                f"parallel ({fp_workers} workers)..."
            )
        try:
            with ProcessPoolExecutor(
                max_workers=fp_workers,
                initializer=_init_scheduling_worker,
                initargs=(instance, _worker_solve_kwargs),
            ) as ex:
                first_pass_solutions = {
                    k: sol for k, sol in ex.map(
                        _solve_driver_in_worker, range(K), cust_lists
                    )
                }
        except Exception as exc:
            warnings.warn(
                f"parallel first pass failed ({type(exc).__name__}: {exc}); "
                "falling back to sequential",
                RuntimeWarning,
                stacklevel=2,
            )
            first_pass_solutions = None

    if first_pass_solutions is None:
        if progress:
            print(f"[scheduling] solving {K} per-driver MILPs (N={N})...")
        first_pass_solutions = {}
        for k in range(K):
            sol_k = _cached_milp(k, cust_lists[k])
            first_pass_solutions[k] = sol_k
            if progress:
                n_cust = len(cust_lists[k])
                n_picked = len(sol_k.get("assigned_customers", []))
                e_sum = float(sol_k.get("total_delay", 0.0))
                e_str = f"{e_sum:.2f}" if e_sum == e_sum else "NaN"
                print(
                    f"  driver {k + 1:>3}/{K}: {n_cust:>3} allowed -> {n_picked:>3} picked  "
                    f"{sol_k['status_name']:<12} sum_e={e_str}  ({sol_k['gurobi_seconds'] * 1000:6.1f}ms)"
                )

    for k in range(K):
        sol_k = first_pass_solutions[k]
        per_driver_base[k] = sol_k
        last_input_set_base[k] = frozenset(cust_lists[k])
        statuses.append(sol_k["status_name"])
        for i in sol_k.get("assigned_customers", []):
            candidates[i].append((k, float(sol_k.get("e", {}).get(i, 0.0))))

    first_pass_wall_seconds = time.perf_counter() - t_first_pass

    t_conflict = time.perf_counter()
    assigned_base: dict[int, int | None] = {}
    e_all_base: dict[int, float] = {}
    n_conflicts = 0
    n_dropped_pairs = 0
    conflicts: dict[int, list[int]] = {}
    conflicts_by_driver: dict[int, int] = {k: 0 for k in range(K)}
    for i in range(N):
        cands = candidates[i]
        if not cands:
            assigned_base[i] = None
            continue
        if len(cands) > 1:
            n_conflicts += 1
            n_dropped_pairs += len(cands) - 1
            conflicts[i] = [k for k, _ in cands]
            for k, _e in cands:
                conflicts_by_driver[k] += 1
        cands.sort(key=lambda kv: (kv[1], kv[0]))
        k_best, e_best = cands[0]
        assigned_base[i] = k_best
        e_all_base[i] = e_best

    total_delay_before_resolve = float(sum(e_all_base.values()))
    total_idle_before_resolve = _aggregate_idle(assigned_base)
    objective_before_resolve = _objective_for(assigned_base, e_all_base)

    if progress:
        print(
            f"[scheduling] greedy conflict resolution: {n_conflicts} customers "
            f"chosen by >1 driver, {n_dropped_pairs} (driver,customer) pairs dropped"
        )

    def _sequential_results() -> dict[str, dict[str, Any]]:
        return {
            s: _run_resolve_strategy(
                s,
                cached_milp=_cached_milp,
                objective_for=_objective_for,
                aggregate_idle=_aggregate_idle,
                assigned_base=assigned_base,
                e_all_base=e_all_base,
                per_driver_base=per_driver_base,
                last_input_set_base=last_input_set_base,
                conflicts_by_driver=conflicts_by_driver,
                driver_to_customers=driver_to_customers,
                K=K,
                max_resolve_rounds=max_resolve_rounds,
                progress=progress,
            )
            for s in STRATEGIES
        }

    resolve_in_parallel = use_parallel and max_resolve_rounds > 0 and len(STRATEGIES) > 1

    strategy_results: dict[str, dict[str, Any]]
    if resolve_in_parallel:
        slim_base = {
            k: {
                "assigned_customers": list(sol.get("assigned_customers", [])),
                "M": float(sol.get("M", time_granularity)),
            }
            for k, sol in per_driver_base.items()
        }
        common = {
            "assigned_base": assigned_base,
            "e_all_base": e_all_base,
            "per_driver_base": slim_base,
            "last_input_set_base": last_input_set_base,
            "conflicts_by_driver": conflicts_by_driver,
            "driver_to_customers": driver_to_customers,
            "max_resolve_rounds": max_resolve_rounds,
        }
        max_workers = min(len(STRATEGIES), os.cpu_count() or 1)
        if progress:
            print(
                f"[scheduling] resolving {len(STRATEGIES)} strategies in "
                f"parallel ({max_workers} workers)..."
            )
        try:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_scheduling_worker,
                initargs=(instance, _worker_solve_kwargs),
            ) as ex:
                futures = {
                    ex.submit(_resolve_strategy_in_worker, {**common, "strategy": s}): s
                    for s in STRATEGIES
                }
                strategy_results = {futures[f]: f.result() for f in futures}
        except Exception as exc:
            warnings.warn(
                f"parallel strategy resolve failed "
                f"({type(exc).__name__}: {exc}); falling back to sequential",
                RuntimeWarning,
                stacklevel=2,
            )
            strategy_results = _sequential_results()
    else:
        strategy_results = _sequential_results()
    conflict_wall_seconds = time.perf_counter() - t_conflict
    chosen_name = min(STRATEGIES, key=lambda s: strategy_results[s]["objective"])
    chosen = strategy_results[chosen_name]
    strategy_objectives = {s: strategy_results[s]["objective"] for s in STRATEGIES}

    if progress:
        obj_str = ", ".join(
            f"{s}={strategy_objectives[s]:.2f}" for s in STRATEGIES
        )
        print(f"[scheduling] strategy objectives: {obj_str}")
        print(
            f"[scheduling] chosen strategy: {chosen_name} "
            f"(obj={chosen['objective']:.2f})"
        )
        total_calls = cache_hits + cache_misses
        hit_rate = (100.0 * cache_hits / total_calls) if total_calls else 0.0
        scope = " (workers use private caches)" if use_parallel else ""
        print(
            f"[scheduling] MILP cache: {cache_hits}/{total_calls} hits "
            f"({hit_rate:.1f}%){scope}"
        )

    assigned = chosen["assigned"]
    e_all = chosen["e_all"]
    per_driver = chosen["per_driver"]
    total_delay = chosen["total_delay"]
    total_idle = chosen["total_idle"]
    objective = chosen["objective"]
    unassigned = [i for i, k in assigned.items() if k is None]
    M_big = float(max((d["M"] for d in per_driver.values()), default=0.0))

    return {
        "status_name": ",".join(statuses) if statuses else "EMPTY",
        "objective": objective,
        "objective_before_resolve": objective_before_resolve,
        "total_delay": float(total_delay),
        "total_delay_before_resolve": total_delay_before_resolve,
        "total_delay_improvement": total_delay_before_resolve - float(total_delay),
        "total_idle": float(total_idle),
        "total_idle_before_resolve": float(total_idle_before_resolve),
        "n_unassigned": len(unassigned),
        "unassigned": unassigned,
        "assigned": assigned,
        "e": e_all,
        "per_driver": per_driver,
        "n_conflicts": n_conflicts,
        "n_dropped_pairs": n_dropped_pairs,
        "n_drivers_resolved": chosen["n_drivers_resolved"],
        "n_drivers_skipped": chosen["n_drivers_skipped"],
        "n_newly_assigned": chosen["n_newly_assigned"],
        "n_resolve_rounds": chosen["n_rounds_run"],
        "conflicts": conflicts,
        "M": M_big,
        "scheduling_wall_seconds": first_pass_wall_seconds,
        "conflict_wall_seconds": conflict_wall_seconds,
        "total_wall_seconds": first_pass_wall_seconds + conflict_wall_seconds,
        "strategy_objectives": strategy_objectives,
        "strategy_resolve_wall_seconds": {
            s: strategy_results[s]["resolve_wall_seconds"] for s in STRATEGIES
        },
        "chosen_strategy": chosen_name,
        "milp_cache_hits": cache_hits,
        "milp_cache_misses": cache_misses,
    }


def attach_job_scheduling_to_pipeline_result(
    result: dict[str, Any],
    *,
    seed: int | None = None,
    E_max: float = 15.0,
    P_penalty: float = 300.0,
    B_min: float = 10.0,
    B_max: float = 60.0,
    I_idle: float = 1.0,
    D_delay: float = 1.0,
    S_k_range: tuple[float, float] = (0.0, 60.0),
    H_k_range: tuple[float, float] = (240.0, 600.0),
    U_range: tuple[float, float] = (0.0, 600.0),
    ride_duration: tuple[float, float] = (5.0, 30.0),
    time_limit: float | None = 30.0,
    mip_gap: float | None = 0.01,
    mip_gap_abs: float | None = None,
    max_resolve_rounds: int = 5,
    time_granularity: float = 1.0,
    progress: bool = True,
) -> dict[str, Any]:
    """Per-driver scheduling on top of the FF matching."""
    matching = result["matching"]
    customer_nodes = matching["customer_nodes"]
    driver_nodes = matching["driver_nodes"]
    N = len(customer_nodes)
    K_eff = len(driver_nodes)
    if K_eff < 1:
        raise ValueError("Scheduling needs at least one driver.")

    if progress:
        print(
            f"[scheduling] generating instance: N={N}, K={K_eff}, "
            f"S~U{S_k_range}, R~U{tuple(ride_duration)}, "
            f"E_max={E_max}, P={P_penalty:g}, "
            f"B_min={B_min:g}, B_max={B_max:g}, I={I_idle:g}, D={D_delay:g}"
        )
    instance = generate_scheduling_instance(
        result,
        seed=seed,
        E_max=E_max,
        P_penalty=P_penalty,
        B_min=B_min,
        B_max=B_max,
        I_idle=I_idle,
        D_delay=D_delay,
        S_k_range=S_k_range,
        H_k_range=H_k_range,
        U_range=U_range,
        ride_duration=ride_duration,
    )
    if progress:
        H_lo = float(instance["H"].min()) if K_eff else 0.0
        H_hi = float(instance["H"].max()) if K_eff else 0.0
        U_lo, U_hi = instance["U_range"]
        print(
            f"[scheduling]   sampled H in [{H_lo:.2f}, {H_hi:.2f}], "
            f"U in [{U_lo:.2f}, {U_hi:.2f}]"
        )
    d_to_c = driver_to_allowed_customers_from_matching(result)

    sol = solve_per_driver_scheduling(
        instance,
        d_to_c,
        time_limit=time_limit,
        mip_gap=mip_gap,
        mip_gap_abs=mip_gap_abs,
        max_resolve_rounds=max_resolve_rounds,
        time_granularity=time_granularity,
        progress=progress,
    )

    H_arr = instance["H"]
    S_arr = instance["S"]
    R_arr = instance["R"]
    U_arr = instance["U"]
    new = dict(result)
    new["scheduling"] = {
        **sol,
        "N_jobs": N,
        "K": K_eff,
        "driver_to_customers": {k: list(v) for k, v in d_to_c.items()},
        "E_max": instance["E_max"],
        "P_penalty": instance["P"],
        "B_min": instance["B_min"],
        "B_max": instance["B_max"],
        "I_idle": instance["I_idle"],
        "D_delay": instance["D_delay"],
        "ride_duration_range": instance["ride_duration_range"],
        "seed": instance["seed"],
        "U": instance["U"].tolist() if hasattr(instance["U"], "tolist") else list(instance["U"]),
        "S": S_arr.tolist(),
        "H": H_arr.tolist(),
        "R": R_arr.tolist() if hasattr(R_arr, "tolist") else list(R_arr),
        "S_min": float(S_arr.min()) if K_eff else 0.0,
        "S_max": float(S_arr.max()) if K_eff else 0.0,
        "S_mean": float(S_arr.mean()) if K_eff else 0.0,
        "H_min": float(H_arr.min()) if K_eff else 0.0,
        "H_max": float(H_arr.max()) if K_eff else 0.0,
        "H_mean": float(H_arr.mean()) if K_eff else 0.0,
        "R_min": float(R_arr.min()) if N else 0.0,
        "R_max": float(R_arr.max()) if N else 0.0,
        "R_mean": float(R_arr.mean()) if N else 0.0,
        "U_min": float(U_arr.min()) if N else 0.0,
        "U_max": float(U_arr.max()) if N else 0.0,
        "U_mean": float(U_arr.mean()) if N else 0.0,
    }
    new["scheduling_instance"] = instance
    return new
