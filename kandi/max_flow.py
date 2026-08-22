"""Ford-Fulkerson max flow, matching extraction and matching metrics."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import networkx as nx
import numpy as np

from kandi.context import CUSTOMER_PART, DRIVER_PART
from kandi.graph_io import road_bipartition


def compute_max_flow(
    G_road: nx.Graph, SOURCE: int, SINK: int
) -> tuple[float, dict[Any, dict[Any, float]], dict[Any, dict[Any, float]], float]:
    """Ford-Fulkerson with BFS for shortest augmenting paths (Edmonds-Karp)."""
    part, _road_nodes = road_bipartition(G_road, SOURCE, SINK)
    cap: dict[Any, dict[Any, float]] = {n: {} for n in G_road.nodes()}
    for u, v, data in G_road.edges(data=True):
        w = float(data["weight"])
        if SOURCE in (u, v):
            a, b = SOURCE, (v if u == SOURCE else u)
        elif SINK in (u, v):
            a, b = (v if u == SINK else u), SINK
        else:
            a, b = (u, v) if part[u] == CUSTOMER_PART else (v, u)
        cap[a][b] = w
    residual: dict[Any, dict[Any, float]] = {n: dict(cap[n]) for n in cap}
    for a in cap:
        for b in cap[a]:
            residual[b].setdefault(a, 0.0)

    EPS = 1e-12
    max_flow = 0.0
    t0 = time.perf_counter()

    while True:
        parent: dict = {SOURCE: None}
        queue: deque = deque([SOURCE])
        found = False
        while queue:
            node = queue.popleft()
            if node == SINK:
                found = True
                break
            for nbr, c in residual[node].items():
                if c > EPS and nbr not in parent:
                    parent[nbr] = node
                    queue.append(nbr)
        if not found:
            break

        bottleneck = float("inf")
        n = SINK
        while parent[n] is not None:
            c = residual[parent[n]][n]
            if c < bottleneck:
                bottleneck = c
            n = parent[n]

        n = SINK
        while parent[n] is not None:
            u = parent[n]
            residual[u][n] -= bottleneck
            residual[n][u] += bottleneck
            n = u

        max_flow += bottleneck

    return max_flow, cap, residual, time.perf_counter() - t0


def extract_customer_driver_flow_matching(
    G_road: nx.Graph,
    cap: dict[Any, dict[Any, float]],
    residual: dict[Any, dict[Any, float]],
    part: dict[int, int],
    road_nodes: list[int],
) -> dict[str, Any]:
    """Read the matching off the residual: matched iff capacity was consumed."""
    customer_nodes = sorted(n for n in road_nodes if part[n] == CUSTOMER_PART)
    driver_nodes = sorted(n for n in road_nodes if part[n] == DRIVER_PART)

    customer_to_driver: dict[int, list[int]] = {c: [] for c in customer_nodes}
    driver_to_customers: dict[int, list[int]] = {d: [] for d in driver_nodes}

    for c in customer_nodes:
        cap_c = cap[c]
        residual_c = residual[c]
        for d in driver_nodes:
            if d not in cap_c:
                continue
            used_flow = cap_c[d] - residual_c[d]
            if used_flow > 1e-9:
                driver_to_customers[d].append(c)
                customer_to_driver[c].append(d)

    unmatched_customers = [c for c, dlist in customer_to_driver.items() if not dlist]
    return {
        "customer_to_driver": customer_to_driver,
        "driver_to_customers": driver_to_customers,
        "unmatched_customers": unmatched_customers,
        "customer_nodes": customer_nodes,
        "driver_nodes": driver_nodes,
    }


def postprocess_matching_metrics(
    matching: dict[str, Any],
    ford_fulkerson_seconds: float,
) -> dict[str, Any]:
    """Mean customers per driver, mean drivers per customer, coverage %, FF time."""
    customer_nodes = matching["customer_nodes"]
    driver_nodes = matching["driver_nodes"]
    customer_to_driver = matching["customer_to_driver"]
    driver_to_customers = matching["driver_to_customers"]
    unmatched = matching["unmatched_customers"]

    n_c = len(customer_nodes)
    n_d = len(driver_nodes)
    coverage_pct = (100.0 * (n_c - len(unmatched)) / n_c) if n_c else 0.0
    avg_customers_per_driver = (
        float(np.mean([len(driver_to_customers[d]) for d in driver_nodes]))
        if n_d
        else 0.0
    )
    avg_drivers_per_customer = (
        float(np.mean([len(customer_to_driver[c]) for c in customer_nodes]))
        if n_c
        else 0.0
    )
    return {
        "ford_fulkerson_seconds": ford_fulkerson_seconds,
        "n_customers": n_c,
        "n_drivers": n_d,
        "avg_customers_per_driver": avg_customers_per_driver,
        "avg_drivers_per_customer": avg_drivers_per_customer,
        "coverage_pct": coverage_pct,
        "n_served_customers": n_c - len(unmatched),
        "n_unmatched_customers": len(unmatched),
    }


def print_flow_matching(m: dict[str, Any], G_road: nx.Graph) -> None:
    customer_to_driver = m["customer_to_driver"]
    customer_nodes = m["customer_nodes"]
    driver_nodes = m["driver_nodes"]
    driver_to_customers = m["driver_to_customers"]

    print("Customer -> drivers mapping:")
    for c in customer_nodes:
        dlist = customer_to_driver[c]
        if not dlist:
            print(f"  {c} -> (none)")
        else:
            parts = []
            for d in sorted(dlist):
                pref = G_road.edges[c, d]["weight"]
                parts.append(f"{d} (weight {pref:g})")
            print(f"  {c} -> " + ", ".join(parts))

    if m["unmatched_customers"]:
        print("\nCustomers with no driver:", m["unmatched_customers"])
    else:
        print("\nEvery customer has at least one driver.")

    print("\nDriver -> customers mapping:")
    for d in driver_nodes:
        print(f"  {d} -> {driver_to_customers[d]}")


