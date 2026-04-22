---
name: gpu-trace-analysis
description: Analyze GPU traces step by step and turn them into prioritized optimization actions.
compatibility: opencode
metadata:
  audience: performance-engineering
  workflow: trace-analysis
---

## What I do

I analyze a `.gputrace`.
I follow a fixed 6-step reasoning process:

1. Establish the baseline inference window.
2. Identify the critical path.
3. Measure time attribution.
4. Correlate trace events with model architecture.
5. Identify inefficiency patterns.
6. Prioritize and hypothesize optimizations.

Use me when you want actionable performance insights from trace data, especially in this repository after `capture_gpu_trace` has been run.

## Repo-Specific Guidance

- If no trace exists yet, start by using `capture_gpu_trace`.
- Begin with `trace_open` to summarize the `.gputrace` bundle structure.
- Use `trace_search` to locate kernel names, labels, signposts, and repeated operation names in plist, json, sqlite, or binary string data.
- If `trace_open` reports SQLite databases inside the bundle, use `trace_sqlite_query` for targeted read-only inspection.

## Workflow

### 1. Establish the Baseline

Identify the exact inference window you care about.

Locate the region using one or more of:

- the target process name
- known function names
- signposts
- repeated kernel groups corresponding to one inference call

Record:

- inference start timestamp
- inference end timestamp
- total duration in ms

Example:

```text
Inference window:
start = 12.184 s
end   = 12.226 s
total = 42.0 ms
```

This window is the optimization target. All other measurements are relative to it.

### 2. Identify the Critical Path

Determine whether total runtime is dominated by GPU execution, CPU-side submission/setup delays, or both.

Compute or estimate:

- total GPU execution time in the inference window
- GPU idle gaps between kernel groups
- CPU activity around command submission
- whether kernels are tightly packed or separated by waits

Rules of thumb:

- If GPU activity occupies most of the window, classify it as GPU-bound.
- If kernels are short and separated by meaningful idle gaps while CPU stays active, classify it as CPU-bound or submission-bound.
- If both are substantial, classify it as mixed.

Instead of visually checking track density, compute:

- sum of GPU kernel durations
- union of GPU busy intervals when possible
- total gap time between kernels or encoder groups

Example:

```text
Total inference time: 42 ms
Total GPU active time: 38 ms
GPU idle gaps inside window: 4 ms
Conclusion: primarily GPU-bound, with some CPU submission overhead
```

### 3. Measure Time Attribution

Group events by operation or kernel name and sum total duration.

For GPU-bound cases:

- group GPU events by kernel or shader name
- sort by total time descending
- report count, total duration, average duration, and percent of inference time

For CPU-bound cases:

- group CPU-side functions, stacks, or categories
- identify where submission/setup overhead is going

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

Optimize by time share, not by event count alone.

### 4. Correlate Trace Events With Model Architecture

Compare observed event counts with expected model operations.

Count major kernel categories such as:

- matmul
- softmax
- layer norm
- add
- copy or blit
- attention-related kernels

Then compare observed counts with what the model architecture should roughly produce.

Example:

```python
expected_matmuls = 6 * (3 + 2)  # 6 layers, 3 attention matmuls, 2 FFN matmuls
# = 30
actual_matmuls = 36
```

Interpretation:

- `actual ~= expected`: behavior is plausible
- `actual >> expected`: extra fragmentation, unfused ops, or repeated passes
- `actual << expected`: some work may already be fused

This comparison is often easier in scripts than by visual inspection.

### 5. Identify Inefficiency Patterns

Detect anti-patterns from the event stream.

Pattern 1: Fragmented command buffers

Signals:

- many short kernels
- repeated small gaps between kernel batches
- high kernel count with low average duration

Heuristic:

- If many kernels are under `1 ms` and separated by frequent `0.1-1 ms` gaps, suspect submission or fragmentation overhead.

Pattern 2: Memory transfer bottlenecks

Signals:

- frequent blit, copy, or transfer events
- transfer time is a meaningful fraction of total runtime

Pattern 3: Imbalanced kernel duration

Signal:

- one kernel family consumes more than `40%` of total runtime

Pattern 4: CPU-GPU synchronization points

Signals:

- GPU idle intervals align with CPU waits, sync calls, or result reads
- repeated host-side pauses before more GPU work is submitted

These patterns show up as counts, grouped durations, gap statistics, and event-sequence structure.

### 6. Prioritize and Hypothesize

Turn the measurements into an optimization plan.

Use this prioritization framework:

- high impact, low effort
- high impact, medium effort
- medium impact, medium effort
- low impact, high effort

Example hypothesis:

```text
Bottleneck: fragmented matmul-heavy execution
Evidence:
- matmul kernels = 24 ms / 42 ms total (57%)
- 36 matmul launches vs 30 expected
- repeated 0.5-1.0 ms gaps between kernel groups
Expected improvement:
- 10-25% from reducing launch fragmentation and using fused ops
Proposed fix:
- replace manual attention sequence with fused MLX attention path
- reduce eager sync points
- batch operations where possible
```

## Decision Tree

Ask these questions in order:

1. Does GPU active time occupy most of the inference window?
2. Does one kernel family consume more than `40%` of total time?
3. Are there many small kernels?
4. Are there frequent GPU idle gaps?

Interpretation:

- Yes to 1: GPU-bound
- No to 1: CPU-bound or mixed
- Yes to 2: optimize that operation first
- Yes to 3: consider fusion, batching, or graph compilation
- Yes to 4: investigate CPU overhead or synchronization


## Output Template

Use this exact structure when reporting results:

```markdown
# CLI Trace Analysis: [Model Name] - [Date]

## 1. Baseline
- Inference start: ___
- Inference end: ___
- Total inference time: ___ ms
- Hardware: ___
- Trace source: [export/json/csv/log]

## 2. Critical Path
- GPU active time: ___ ms
- GPU idle gap time: ___ ms
- CPU submission/setup time: ___ ms
- Classification:
  - [ ] GPU-bound
  - [ ] CPU-bound
  - [ ] Mixed

## 3. Time Attribution
Top operations by total time:

| Operation | Count | Total ms | Avg ms | % of inference |
|----------|------:|---------:|-------:|---------------:|
|          |       |          |        |                |
|          |       |          |        |                |
|          |       |          |        |                |

## 4. Architecture Correlation
- Expected operation counts: ___
- Observed operation counts: ___
- Interpretation: ___

## 5. Inefficiency Patterns
- Fragmentation signals: ___
- Transfer overhead: ___
- Synchronization points: ___
- Dominant bottleneck pattern: ___

## 6. Prioritized Actions
1. ___
2. ___
3. ___

## Evidence-Based Hypothesis
- Bottleneck: ___
- Evidence: ___
- Expected improvement: ___
- Proposed fix: ___
```

## Working Style

- Be quantitative and specific.
- Prefer direct measurements over guesses.
- If exact values are unavailable, state that clearly and provide the best bounded estimate you can.
- Tie every recommendation back to evidence from the trace.
