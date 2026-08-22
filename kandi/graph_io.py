"""Loading bipartite ride-sharing graphs from ``.graph`` files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

UNIT_EDGE_WEIGHT = 1.0


def read_text_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def parse_graph_id(header_line: str) -> int:
    parts = header_line.strip().split()
    if len(parts) < 3 or parts[0] != "#" or parts[1].lower() != "graph":
        raise ValueError(f"Expected '# graph <id>', got: {header_line!r}")
    return int(parts[2])


def parse_edge_line(line: str) -> tuple[int, int, float]:
    u, v, w = line.strip().split()
    return int(u), int(v), float(w)


def iter_graph_records(lines: list[str]):
    """Yield (graph_id, num_nodes, edges) per ``# graph`` block."""
    i = 0
    nlines = len(lines)
    while i < nlines:
        line = lines[i].strip()
        if not line.startswith("# graph"):
            i += 1
            continue
        gid = parse_graph_id(lines[i])
        i += 1
        if i >= nlines:
            raise ValueError(f"Missing node count after graph {gid}")
        num_nodes = int(lines[i].strip())
        i += 1
        edges = []
        while i < nlines:
            nxt = lines[i].strip()
            if nxt.startswith("# graph"):
                break
            if nxt:
                edges.append(parse_edge_line(lines[i]))
            i += 1
        yield gid, num_nodes, edges


def weighted_graph_from_edges(edges: list[tuple[int, int, float]]) -> nx.Graph:
    G = nx.Graph()
    for u, v, _w in edges:
        G.add_edge(u, v, weight=UNIT_EDGE_WEIGHT)
    return G


def attach_bipartite_source_sink(
    G: nx.Graph,
    *,
    customer_capacity: float = UNIT_EDGE_WEIGHT,
    driver_capacity: float | None = None,
) -> tuple[int, int]:
    """Add super-source/sink; ``driver_capacity=None`` means unlimited."""
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot attach terminals to an empty graph.")
    color = nx.bipartite.color(G)
    if 0 in G and color[0] == 1:
        color = {n: 1 - c for n, c in color.items()}

    n_customers = sum(1 for n in G.nodes() if color[n] == 0)
    if driver_capacity is None:
        driver_cap = float(max(n_customers, 1))
    else:
        driver_cap = float(driver_capacity)
    cust_cap = float(customer_capacity)

    base = max(G.nodes())
    super_s = base + 1
    super_t = base + 2

    for n in list(G.nodes()):
        if color[n] == 0:
            G.add_edge(super_s, n, weight=cust_cap)
        else:
            G.add_edge(n, super_t, weight=driver_cap)

    return super_s, super_t


def road_bipartition(
    G_road: nx.Graph, source: int, sink: int
) -> tuple[dict[int, int], list[int]]:
    """Road-only 2-coloring: 0 = customers, 1 = drivers."""
    road_nodes = [n for n in G_road.nodes() if n not in (source, sink)]
    G_core = G_road.subgraph(road_nodes).copy()
    part = nx.bipartite.color(G_core)
    if 0 in G_core and part[0] == 1:
        part = {n: 1 - c for n, c in part.items()}
    return part, road_nodes


def build_graph_entry(
    graph_id: int, num_nodes: int, edges: list[tuple[int, int, float]]
) -> dict:
    G = weighted_graph_from_edges(edges)
    return {
        "id": graph_id,
        "num_nodes": num_nodes,
        "graph": G,
        "num_edges": G.number_of_edges(),
        "num_nodes_in_graph": G.number_of_nodes(),
    }


def load_bipartite_graphs_from_file(
    graph_path: Path,
    *,
    customer_capacity: float = UNIT_EDGE_WEIGHT,
    driver_capacity: float | None = None,
) -> list[dict]:
    """Load a .graph file, drop non-bipartite graphs, attach source/sink."""
    graphs: list[dict] = []
    for gid, num_nodes, edges in iter_graph_records(read_text_lines(graph_path)):
        graphs.append(build_graph_entry(gid, num_nodes, edges))

    print(f"Loaded {len(graphs)} graphs from {graph_path.resolve()}")

    non_bipartite_ids = [e["id"] for e in graphs if not nx.is_bipartite(e["graph"])]
    graphs = [e for e in graphs if nx.is_bipartite(e["graph"])]

    for entry in graphs:
        s, t = attach_bipartite_source_sink(
            entry["graph"],
            customer_capacity=customer_capacity,
            driver_capacity=driver_capacity,
        )
        entry["source"] = s
        entry["sink"] = t
        entry["num_edges"] = entry["graph"].number_of_edges()
        entry["num_nodes_in_graph"] = entry["graph"].number_of_nodes()

    print(
        f"Kept {len(graphs)} bipartite graphs; "
        f"discarded {len(non_bipartite_ids)} non-bipartite (ids: {non_bipartite_ids})"
    )
    print(
        "Each kept graph: super-source (customers, partition 0) and super-sink (drivers, partition 1) added; "
        "entry['source'] / entry['sink'] are those terminal ids."
    )
    return graphs


def generate_complex_bipartite_graph_lines(
    *,
    graph_id: int,
    n_customers: int,
    n_drivers: int,
    edges_per_customer_min: int = 3,
    edges_per_customer_max: int = 6,
    seed: int = 0,
) -> list[str]:
    """Return ``# graph`` lines; nodes 0..N_c-1 customers, the rest drivers."""
    rng = np.random.default_rng(seed)
    n_total = n_customers + n_drivers
    cust_ids = list(range(n_customers))
    drv_ids = list(range(n_customers, n_total))
    edges: set[tuple[int, int]] = set()
    for c in cust_ids:
        k = int(rng.integers(edges_per_customer_min, edges_per_customer_max + 1))
        k = min(k, n_drivers)
        chosen = rng.choice(drv_ids, size=k, replace=False)
        for d in chosen:
            edges.add((c, int(d)))
    for d in drv_ids:
        if not any((c, d) in edges for c in cust_ids):
            c = int(rng.choice(cust_ids))
            edges.add((c, d))
    lines = [f"# graph {graph_id}", str(n_total)]
    for u, v in sorted(edges):
        lines.append(f"{u} {v} 1")
    return lines


def write_complex_bipartite_graph_file(
    path: Path, *, configs: list[dict[str, Any]]
) -> Path:
    """Write one ``# graph`` block per config (kwargs for the generator)."""
    all_lines: list[str] = []
    for cfg in configs:
        all_lines.extend(generate_complex_bipartite_graph_lines(**cfg))
    path.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    return path
