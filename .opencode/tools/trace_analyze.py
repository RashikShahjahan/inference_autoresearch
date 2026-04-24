from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Iterable

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from trace_query_xpath import _tool_result, export_trace_xpath, resolve_repo_path

NS_PER_MS = 1_000_000


@contextlib.contextmanager
def _buffer_stderr_on_success():
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        try:
            yield
        except Exception:
            sys.stderr.write(buffer.getvalue())
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a Metal trace into a compact inference bottleneck summary"
    )
    parser.add_argument(
        "--trace-path",
        default="state/batch_generate_profile.trace",
        help="Path to a .trace document, relative to repo root or absolute",
    )
    parser.add_argument(
        "--run-number",
        type=int,
        default=1,
        help="Trace run number to analyze",
    )
    parser.add_argument(
        "--process-name",
        default="python3",
        help="Process name filter for inference rows",
    )
    parser.add_argument(
        "--cluster-gap-ms",
        type=float,
        default=500.0,
        help="Start-to-start gap threshold used to split warmup and measured clusters",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top grouped operations to report",
    )
    return parser


def export_table(trace_path: Path, schema: str, run_number: int) -> tuple[list[str], list[dict[str, dict[str, str]]]]:
    xpath = f'/trace-toc/run[@number="{run_number}"]/data/table[@schema="{schema}"]'
    payload = export_trace_xpath(trace_path, xpath)
    root = ET.fromstring(payload["export_xml"])
    node = root.find("node")
    if node is None:
        return [], []

    schema_node = node.find("schema")
    columns: list[str] = []
    if schema_node is not None:
        for col in schema_node.findall("col"):
            mnemonic = col.findtext("mnemonic")
            columns.append(mnemonic or f"column_{len(columns)}")

    rows: list[dict[str, dict[str, str]]] = []
    cell_by_id: dict[str, dict[str, str]] = {}
    for row_node in node.findall("row"):
        row: dict[str, dict[str, str]] = {}
        for index, child in enumerate(list(row_node)):
            key = columns[index] if index < len(columns) else child.tag
            ref = child.attrib.get("ref")
            if ref is not None and ref in cell_by_id:
                row[key] = dict(cell_by_id[ref])
                continue

            raw = child.text.strip() if child.text and child.text.strip() else ""
            fmt = child.attrib.get("fmt", "")
            row[key] = {"raw": raw, "fmt": fmt or raw, "tag": child.tag}
            cell_id = child.attrib.get("id")
            if cell_id is not None:
                cell_by_id[cell_id] = dict(row[key])
        rows.append(row)
    return columns, rows


def cell_value(row: dict[str, dict[str, str]], key: str) -> str:
    cell = row.get(key)
    if cell is None:
        return ""
    return cell.get("fmt") or cell.get("raw") or ""


def cell_ns(row: dict[str, dict[str, str]], key: str) -> int | None:
    cell = row.get(key)
    if cell is None:
        return None
    raw = cell.get("raw", "").strip().replace(",", "")
    if not raw or not raw.lstrip("-").isdigit():
        return None
    return int(raw)


def matches_process(row: dict[str, dict[str, str]], process_name: str) -> bool:
    if not process_name:
        return True
    target = process_name.lower()
    return any(target in (cell.get("fmt") or cell.get("raw") or "").lower() for cell in row.values())


def fmt_ms(ns: int | float | None) -> float | None:
    if ns is None:
        return None
    return round(float(ns) / NS_PER_MS, 3)


def fmt_time(ns: int | None) -> str | None:
    if ns is None:
        return None
    total_us = round(ns / 1_000)
    minutes, rem_us = divmod(total_us, 60_000_000)
    seconds, rem_us = divmod(rem_us, 1_000_000)
    millis, micros = divmod(rem_us, 1_000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}.{micros:03d}"


def row_end_ns(row: dict[str, dict[str, str]]) -> int | None:
    start = cell_ns(row, "start")
    if start is None:
        start = cell_ns(row, "timestamp")
    duration = cell_ns(row, "duration") or 0
    return None if start is None else start + duration


def row_start_ns(row: dict[str, dict[str, str]]) -> int | None:
    return cell_ns(row, "start") or cell_ns(row, "timestamp")


def rows_in_window(rows: Iterable[dict[str, dict[str, str]]], start_ns: int, end_ns: int) -> list[dict[str, dict[str, str]]]:
    selected = []
    for row in rows:
        start = row_start_ns(row)
        end = row_end_ns(row)
        if start is None or end is None:
            continue
        if start < end_ns and end > start_ns:
            selected.append(row)
    return selected


def build_clusters(rows: list[dict[str, dict[str, str]]], gap_ns: int) -> list[dict]:
    timed_rows = [(row_start_ns(row), row_end_ns(row), row) for row in rows]
    timed_rows = [(s, e, row) for s, e, row in timed_rows if s is not None and e is not None]
    timed_rows.sort(key=lambda item: item[0])
    if not timed_rows:
        return []

    clusters = []
    start, end, _row = timed_rows[0]
    previous_start = start
    count = 1
    largest_gap = 0
    for row_start, row_end, _row in timed_rows[1:]:
        gap = row_start - previous_start
        if gap > gap_ns:
            clusters.append(
                {
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": end - start,
                    "row_count": count,
                    "largest_internal_start_gap_ns": largest_gap,
                }
            )
            start = row_start
            end = row_end
            count = 1
            largest_gap = 0
        else:
            end = max(end, row_end)
            count += 1
            largest_gap = max(largest_gap, gap)
        previous_start = row_start

    clusters.append(
        {
            "start_ns": start,
            "end_ns": end,
            "duration_ns": end - start,
            "row_count": count,
            "largest_internal_start_gap_ns": largest_gap,
        }
    )
    return clusters


def compact_cluster(cluster: dict) -> dict:
    return {
        "start": fmt_time(cluster["start_ns"]),
        "end": fmt_time(cluster["end_ns"]),
        "duration_ms": fmt_ms(cluster["duration_ns"]),
        "row_count": cluster["row_count"],
        "largest_internal_start_gap_ms": fmt_ms(cluster["largest_internal_start_gap_ns"]),
    }


def union_duration(intervals: list[tuple[int, int]]) -> tuple[int, int, list[int]]:
    if not intervals:
        return 0, 0, []
    intervals.sort()
    merged: list[tuple[int, int]] = []
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))

    active = sum(end - start for start, end in merged)
    gaps = [merged[i][0] - merged[i - 1][1] for i in range(1, len(merged))]
    return active, len(merged), gaps


def number_stats(ns_values: list[int]) -> dict:
    if not ns_values:
        return {"count": 0}
    values = sorted(ns_values)
    p95 = values[int(0.95 * (len(values) - 1))]
    return {
        "count": len(values),
        "sum_ms": fmt_ms(sum(values)),
        "avg_ms": fmt_ms(sum(values) / len(values)),
        "median_ms": fmt_ms(statistics.median(values)),
        "p95_ms": fmt_ms(p95),
        "max_ms": fmt_ms(max(values)),
    }


def start_gap_stats(rows: list[dict[str, dict[str, str]]]) -> dict:
    starts = sorted(start for row in rows if (start := row_start_ns(row)) is not None)
    gaps = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
    if not gaps:
        return {"count": 0}
    return {
        "count": len(gaps),
        "avg_ms": fmt_ms(sum(gaps) / len(gaps)),
        "median_ms": fmt_ms(statistics.median(gaps)),
        "p95_ms": fmt_ms(sorted(gaps)[int(0.95 * (len(gaps) - 1))]),
        "max_ms": fmt_ms(max(gaps)),
        "gaps_over_10ms": sum(gap > 10 * NS_PER_MS for gap in gaps),
        "gaps_over_50ms": sum(gap > 50 * NS_PER_MS for gap in gaps),
        "largest_gaps_ms": [fmt_ms(gap) for gap in sorted(gaps, reverse=True)[:10]],
    }


def group_by_label(rows: list[dict[str, dict[str, str]]], preferred_keys: list[str], top_n: int, window_ns: int) -> list[dict]:
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "duration_ns": 0})
    for row in rows:
        label = ""
        for key in preferred_keys:
            label = cell_value(row, key).strip()
            if label:
                break
        if not label:
            label = "unlabeled"
        label = normalize_label(label)
        duration = cell_ns(row, "duration") or 0
        groups[label]["count"] += 1
        groups[label]["duration_ns"] += duration

    ranked = sorted(groups.items(), key=lambda item: item[1]["duration_ns"], reverse=True)
    result = []
    for label, stats in ranked[:top_n]:
        duration_ns = stats["duration_ns"]
        count = stats["count"]
        result.append(
            {
                "label": label,
                "count": count,
                "total_ms": fmt_ms(duration_ns),
                "avg_ms": fmt_ms(duration_ns / count if count else 0),
                "percent_of_window": round(100 * duration_ns / window_ns, 2) if window_ns else 0.0,
            }
        )
    return result


def normalize_label(label: str) -> str:
    label = re.sub(r"\s+0x[0-9a-fA-F]+\s*$", "", label)
    label = re.sub(r"Frame\s+[0-9,]+", "Frame N", label)
    return re.sub(r"\s+", " ", label).strip()


def analyze(args: argparse.Namespace) -> dict:
    if args.run_number <= 0:
        raise ValueError("run_number must be positive")
    if args.cluster_gap_ms <= 0:
        raise ValueError("cluster_gap_ms must be positive")
    if args.top_n <= 0:
        raise ValueError("top_n must be positive")

    trace_path = resolve_repo_path(args.trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace path does not exist: {trace_path}")

    gpu_columns, gpu_rows = export_table(trace_path, "metal-gpu-intervals", args.run_number)
    submission_columns, submission_rows = export_table(
        trace_path,
        "metal-application-command-buffer-submissions",
        args.run_number,
    )
    app_columns, app_rows = export_table(trace_path, "metal-application-intervals", args.run_number)
    gpu_info_columns, gpu_info_rows = export_table(trace_path, "metal-gpu-info", args.run_number)

    process_gpu_rows = [row for row in gpu_rows if matches_process(row, args.process_name)]
    process_submission_rows = [row for row in submission_rows if matches_process(row, args.process_name)]
    process_app_rows = [row for row in app_rows if matches_process(row, args.process_name)]

    gap_ns = int(args.cluster_gap_ms * NS_PER_MS)
    cluster_source = process_submission_rows or process_gpu_rows or process_app_rows
    clusters = build_clusters(cluster_source, gap_ns)
    if not clusters:
        raise RuntimeError("No timestamped trace rows matched the requested process filter")
    measured_cluster = clusters[-1]
    window_start = measured_cluster["start_ns"]
    window_end = measured_cluster["end_ns"]
    window_ns = window_end - window_start

    window_gpu_rows = rows_in_window(process_gpu_rows, window_start, window_end)
    window_submission_rows = rows_in_window(process_submission_rows, window_start, window_end)
    window_app_rows = rows_in_window(process_app_rows, window_start, window_end)

    gpu_intervals = []
    summed_gpu_ns = 0
    for row in window_gpu_rows:
        start = row_start_ns(row)
        end = row_end_ns(row)
        if start is None or end is None:
            continue
        clipped_start = max(start, window_start)
        clipped_end = min(end, window_end)
        if clipped_end <= clipped_start:
            continue
        gpu_intervals.append((clipped_start, clipped_end))
        summed_gpu_ns += clipped_end - clipped_start

    active_ns, active_segments, idle_gaps = union_duration(gpu_intervals)
    idle_ns = max(0, window_ns - active_ns)
    active_share = active_ns / window_ns if window_ns else 0.0
    if active_share >= 0.85 and idle_ns < 0.15 * window_ns:
        classification = "gpu-bound"
    elif active_share <= 0.5:
        classification = "cpu/submission-bound"
    else:
        classification = "mixed"

    command_durations = [duration for row in window_submission_rows if (duration := cell_ns(row, "duration")) is not None]
    encoder_times = [duration for row in window_submission_rows if (duration := cell_ns(row, "encoder-time")) is not None]

    gpu_info = []
    for row in gpu_info_rows:
        gpu_info.append(
            {
                "timestamp": cell_value(row, "timestamp"),
                "gpu_name": cell_value(row, "gpu-name"),
                "gpu_index": cell_value(row, "gpu-index"),
            }
        )

    issues = []
    if idle_ns > 0:
        issues.append(
            f"GPU idle gaps account for {fmt_ms(idle_ns)} ms ({round(100 * idle_ns / window_ns, 2)}%) of the inferred measured window."
        )
    gap_summary = start_gap_stats(window_submission_rows)
    if gap_summary.get("gaps_over_10ms", 0):
        issues.append(
            f"Command submissions have {gap_summary['gaps_over_10ms']} start gaps over 10 ms and {gap_summary.get('gaps_over_50ms', 0)} over 50 ms."
        )
    if window_gpu_rows:
        issues.append(
            f"The measured window contains {len(window_gpu_rows)} GPU intervals for the target process; reducing per-token command count is a trace-supported target."
        )

    return {
        "trace_path": str(trace_path),
        "run_number": args.run_number,
        "process_name": args.process_name,
        "gpu_info": gpu_info,
        "schemas": {
            "metal-gpu-intervals": {
                "columns": gpu_columns,
                "row_count": len(gpu_rows),
                "process_row_count": len(process_gpu_rows),
            },
            "metal-application-command-buffer-submissions": {
                "columns": submission_columns,
                "row_count": len(submission_rows),
                "process_row_count": len(process_submission_rows),
            },
            "metal-application-intervals": {
                "columns": app_columns,
                "row_count": len(app_rows),
                "process_row_count": len(process_app_rows),
            },
            "metal-gpu-info": {
                "columns": gpu_info_columns,
                "row_count": len(gpu_info_rows),
            },
        },
        "measured_window": {
            "start": fmt_time(window_start),
            "end": fmt_time(window_end),
            "duration_ms": fmt_ms(window_ns),
            "selection": "last dense target-process cluster after splitting by start gaps",
            "cluster_gap_ms": args.cluster_gap_ms,
            "clusters": [compact_cluster(cluster) for cluster in clusters],
        },
        "critical_path": {
            "classification": classification,
            "gpu_active_ms": fmt_ms(active_ns),
            "gpu_summed_interval_ms": fmt_ms(summed_gpu_ns),
            "gpu_idle_gap_ms": fmt_ms(idle_ns),
            "gpu_active_percent": round(100 * active_share, 2) if window_ns else 0.0,
            "gpu_idle_percent": round(100 * idle_ns / window_ns, 2) if window_ns else 0.0,
            "gpu_interval_count": len(window_gpu_rows),
            "gpu_active_segment_count": active_segments,
            "largest_gpu_idle_gaps_ms": [fmt_ms(gap) for gap in sorted(idle_gaps, reverse=True)[:10]],
        },
        "gpu_operations": group_by_label(
            window_gpu_rows,
            ["event-label", "event-type", "channel-name", "channel-subtitle", "process"],
            args.top_n,
            window_ns,
        ),
        "command_submissions": {
            "row_count": len(window_submission_rows),
            "start_gap_stats": gap_summary,
            "duration_stats": number_stats(command_durations),
            "encoder_time_stats": number_stats(encoder_times),
        },
        "application_intervals": {
            "row_count": len(window_app_rows),
            "top_labels": group_by_label(
                window_app_rows,
                ["event-label", "event-type", "process"],
                args.top_n,
                window_ns,
            ),
        },
        "trace_observed_issues": issues,
        "limitations": [
            "The exported GPU interval tables expose command labels, not MLX kernel names, so this summary cannot rank individual kernels by duration."
        ],
    }


def main() -> int:
    args = build_parser().parse_args()
    with _buffer_stderr_on_success():
        result = analyze(args)
        print(json.dumps(_tool_result(result), indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
