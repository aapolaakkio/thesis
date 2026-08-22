# Ride-sharing scheduling pipeline

Assigning customers to drivers and building each driver's daily schedule. The repository holds both the implementation (`kandi/`) and the thesis source (`main.tex`, `sections/`).

The problem is split into three subproblems, solved in sequence:

1. **Matching** — customers state which drivers they will ride with. The preference graph is bipartite, so a max-flow computation (Ford-Fulkerson with BFS augmenting paths, i.e. Edmonds-Karp) yields an assignable customer set `A_k` for every driver, subject to per-side capacities.
2. **Scheduling** — one time-indexed MILP per driver (Gurobi), over that driver's `A_k`. Minimizes pickup delay, unserved customers and idle time, subject to the shift window, a delay cap, and minimum/maximum breaks between consecutive rides.
3. **Conflict resolution** — the per-driver MILPs run independently, so a customer can be picked by several drivers. A greedy pass keeps the driver with the lowest delay, then the schedules are re-solved iteratively. Five driver orderings are tried and the lowest total objective wins.

Distances are not modelled: a ride occupies its duration `R_i`, and travel between drop-off and the next pickup is approximated by the minimum break `B_min`.

## Requirements

- Python 3.13 (the pinned versions in `requirements.txt` are what the results were produced with)
- A Gurobi license — the free size-limited license is not enough for the instances in the thesis; academic licenses are free

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running

`pipeline.ipynb` is a thin demo notebook; the implementation lives in
`kandi/`, so edit the package and re-run the cells.

```python
from kandi import run_complex_test, run_complex_test_sweep

# One generated instance, end to end.
result = run_complex_test(n_customers=2000, n_drivers=100, customer_capacity=5.0)

# The thesis sweep: 18 scenarios plus one reference instance.
results = run_complex_test_sweep(
    pairs=[(1000, 200)],
    customer_range=(6000, 8000, 1000),
    driver_range=(100, 150, 10),
    edges_per_customer=(3, 6),
    customer_capacity=5.0,
    driver_capacity=None,
)
```

Each run generates its graph into `graphs/` and writes per-instance metrics to a timestamped CSV in `metrics/`. Both directories are gitignored.

Large instances (`N >= 1000`) automatically solve the per-driver MILPs and the five resolve strategies across processes. Pass `parallel=False` to `solve_per_driver_scheduling` to force sequential execution

## Figures

Each script in `figures/` writes its PDF next to itself:

```bash
.venv/bin/python figures/results_plots.py metrics/<sweep>.csv
```

`results_plots.py` reads a sweep metrics CSV and produces the four result figures

## Layout

```
kandi/                  the pipeline, imported by the notebook
  graph_io.py           .graph file parsing, synthetic instance generation,
                        super-source/sink attachment
  context.py            customer/driver partition + graph drawing
  max_flow.py           Ford-Fulkerson, matching extraction, matching metrics
  scheduling.py         per-driver MILP, greedy resolution, resolve strategies
  pipeline.py           per-graph / batch / sweep entry points
  csv_export.py         metrics CSV schema and writer

figures/                figure generation, each script writes its own PDF
  bipartite_graph.py            preference graph with source and sink
  pipeline_process_flow.py      the three-stage pipeline diagram
  ridesharing_illustration.py   road network with customers and drivers
  results_plots.py              runtime, strategy and objective plots

main.tex                thesis root
sections/               thesis text, one file per section
library.bib             bibliography
aaltothesis.cls         Aalto template (also aaltologo.sty, *.bst, readme.txt)

metrics/                sweep result CSVs (gitignored)
graphs/                 generated instances (gitignored)
```

## Instance format

A `.graph` file holds one or more blocks:

```
# graph <id>
<node count>
<u> <v> <weight>
...
```

Nodes `0..N_c-1` are customers and the rest are drivers; each edge is one customer's preference for one driver. The weight column is parsed but ignored every preference edge counts as one, and the capacities come from the super-source and super-sink edges instead. Non-bipartite graphs are dropped on load.
