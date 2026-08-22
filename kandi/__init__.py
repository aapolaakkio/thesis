"""Customer-driver matching + per-driver scheduling pipeline."""

from __future__ import annotations

from kandi.graph_io import (
    UNIT_EDGE_WEIGHT,
    attach_bipartite_source_sink,
    build_graph_entry,
    generate_complex_bipartite_graph_lines,
    iter_graph_records,
    load_bipartite_graphs_from_file,
    parse_edge_line,
    parse_graph_id,
    read_text_lines,
    road_bipartition,
    weighted_graph_from_edges,
    write_complex_bipartite_graph_file,
)
from kandi.context import (
    CUSTOMER_PART,
    DRIVER_PART,
    build_graph_context,
    draw_road_graph,
    print_graph_context,
)
from kandi.csv_export import (
    PIPELINE_CSV_FIELDS,
    pipeline_result_to_csv_row,
    timestamped_metrics_csv_path,
    write_pipeline_results_csv,
)
from kandi.max_flow import (
    compute_max_flow,
    extract_customer_driver_flow_matching,
    postprocess_matching_metrics,
    print_flow_matching,
)
from kandi.scheduling import (
    attach_job_scheduling_to_pipeline_result,
    driver_to_allowed_customers_from_matching,
    generate_scheduling_instance,
    solve_per_driver_scheduling,
    solve_single_driver_milp,
)
from kandi.pipeline import (
    print_complex_test_summary,
    print_driver_schedules,
    process_all_graphs,
    process_graph_at_index,
    run_complex_test,
    run_complex_test_sweep,
)

