---
name: trace-analysis
description: Analyze Instruments traces step by step and turn them into prioritized optimization actions.
compatibility: opencode
metadata:
  audience: performance-engineering
  workflow: trace-analysis
---

## What I do

I analyze Instruments traces using the repository's trace tools.
I follow a fixed 3-step reasoning process:

1. Identify the critical path.
2. Measure time attribution.
3. Record trace-observed inefficiencies.

Use me when you want actionable performance insights from trace data and you can inspect that trace with the available tools.

## Preconditions

- If benchmark state has not been validated in the current session, use `benchmark_prepare` before capturing a new trace.
- If no `.trace` exists yet, start by using `capture_trace`.


## Command Workflow

1. Use `benchmark_prepare` when the benchmark state is unknown or a fresh workspace is being initialized. Do not call `uv run prepare.py` directly from this skill.

2. Use `capture_trace` when `state/batch_generate_profile.trace` is missing, stale, or not representative of the current `generate.py` candidate.

3. Use `trace_analyze` as the primary trace-analysis tool. Run it with defaults first unless there is a concrete reason to change `trace_path`, `run_number`, `process_name`, `cluster_gap_ms`, or `top_n`.

4. Treat these `trace_analyze` fields as the canonical first-pass evidence:

- `measured_window`: selected measured post-warmup cluster, timing, and cluster split evidence.
- `critical_path`: classification, GPU active time, summed GPU interval time, idle gap time, and active/idle percentages.
- `gpu_operations`: grouped GPU labels by total time.
- `command_submissions`: submission counts, start-gap stats, duration stats, and encoder-time stats.
- `application_intervals`: CPU/application-side interval grouping when available.
- `trace_observed_issues`: directly supported bottlenecks and inefficiencies.
- `limitations`: missing attribution or unsupported conclusions.

5. Use `trace_toc`, `trace_export_table`, and `trace_query_xpath` only if `trace_analyze.limitations` or an optimization question requires deeper evidence than the standard summary provides.

6. If using manual exports, inspect the TOC first, then export only the specific run and schemas needed. Do not guess table schemas.

7. Repeat `trace_analyze` after a new `capture_trace` or after changing `generate.py`; do not reuse old summaries for a changed candidate.

The normal path is `benchmark_prepare` if needed, then `capture_trace` if needed, then `trace_analyze`, then interpret the result. Manual table exports are fallback tools only.

## Workflow

### 1. Identify the Critical Path

Determine whether total runtime is dominated by GPU execution, CPU-side submission/setup delays, or both.

Use the measured post-warmup `batch_generate(...)` call captured by `capture_trace`; do not re-establish a separate baseline window unless the trace contains multiple plausible measured calls.

Read these values from `trace_analyze.critical_path`:

- `classification`
- `gpu_active_ms`
- `gpu_summed_interval_ms`
- `gpu_idle_gap_ms`
- `gpu_active_percent`
- `gpu_idle_percent`
- `largest_gpu_idle_gaps_ms`

Use `trace_analyze.measured_window` to report the selected measured cluster and why it was chosen. Use `trace_analyze.command_submissions.start_gap_stats` to support CPU/submission conclusions.

State the evidence for the classification instead of relying on fixed thresholds alone. If `trace_analyze.limitations` says kernel-level names are unavailable, do not invent kernel attribution.

Example:

```text
Total inference time: 42 ms
Total GPU active time: 38 ms
GPU idle gaps inside window: 4 ms
Conclusion: primarily GPU-bound, with some CPU submission overhead
```

### 2. Measure Time Attribution

Group events by operation or kernel family and sum total duration.

Use `trace_analyze.gpu_operations` as the default operation attribution table.

For GPU-bound cases, report the top `gpu_operations` by total time. If labels are generic command labels, say so and avoid pretending they are kernel names.

For CPU/submission-bound or mixed cases, report `trace_analyze.command_submissions` and `trace_analyze.application_intervals`.

Optimize by time share and trace-supported issue size, not by event count alone.

Preferred table format:

```text
Kernel name         Count   Total ms   Avg ms   Percent
-------------------------------------------------------
matmul_kernel       36      24.0       0.67     57.1%
softmax_kernel      12       6.0       0.50     14.3%
layer_norm_kernel   12       4.0       0.33      9.5%
add_kernel          24       2.0       0.08      4.8%
other               --       6.0       --       14.3%
```

### 3. Record Trace-Observed Inefficiencies

Only include issues that are directly supported by exported trace data.

Start with `trace_analyze.trace_observed_issues`. For each issue you report, include:

- the specific trace evidence
- where it appears in the measured inference window
- why it matters for end-to-end inference time

If `trace_analyze.limitations` says a category is unsupported, say so rather than inferring it.

## Output Template

Use this exact structure when reporting results:

```markdown
# CLI Trace Analysis: [Model Name] - [Date]

## 1. Critical Path
- Trace file: ___
- Exported tables used: ___
- Run number: ___
- Hardware: ___
- Trace source: `trace_analyze` plus `trace_toc`, `trace_export_table`, or `trace_query_xpath` if used
- Total inference time: ___ ms
- GPU active time: ___ ms
- GPU idle gap time: ___ ms
- CPU submission/setup time: ___ ms
- Classification:
  - [ ] GPU-bound
  - [ ] CPU-bound
  - [ ] Mixed

## 2. Time Attribution
Top operations by total time:

| Operation | Count | Total ms | Avg ms | % of inference |
|----------|------:|---------:|-------:|---------------:|
|          |       |          |        |                |
|          |       |          |        |                |
|          |       |          |        |                |

## 3. Trace-Observed Issues
- first issue
- next issue
```

## Working Style

- Use `trace_analyze` first for the standard inference trace summary.
- Use the `trace_toc` tool for TOC export, then `trace_export_table` for common table exports only when deeper manual inspection is needed.
- Use `trace_query_xpath` only when a schema-level export is not specific enough.
- For manual exports, start with the TOC, then export only the relevant tables.
- Be quantitative and specific.
- Prefer direct measurements over guesses.
- If the trace tools fail or the input is not a `.trace` file, state the blocker immediately.
