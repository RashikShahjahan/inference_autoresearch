---
name: gpu-trace-analysis
description: Analyze GPU traces step by step and turn them into prioritized optimization actions.
compatibility: opencode
metadata:
  audience: performance-engineering
  workflow: trace-analysis
---

## What I do

I analyze Apple GPU and Metal traces using `xctrace` through the `bash` tool.
I follow a fixed 6-step reasoning process:

1. Establish the baseline inference window.
2. Identify the critical path.
3. Measure time attribution.
4. Correlate trace events with model architecture.
5. Identify inefficiency patterns.
6. Prioritize and hypothesize optimizations.

Use me when you want actionable performance insights from trace data and you can inspect that trace with `xctrace`.

## Preconditions

- `xctrace` must be runnable from `bash`. This requires full Xcode, not just Command Line Tools.
- If Command Line Tools is the active developer directory, use `DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"` for direct `xctrace` commands.
- Preferred input is an Instruments `.trace` document. `xctrace export` is documented for `.trace` files.
- If no `.trace` exists yet, start by using `capture_gpu_trace`.

## Repo-Specific Guidance

- Use the `bash` tool to run `xctrace` directly.
- Start with the `trace_toc` tool to inspect the trace contents before making any claims.
- Export only the tables needed for analysis with `xctrace export --xpath ...`.
- Keep the work focused on analysis: measure the bottleneck, explain it, and turn it into concrete `generate.py` optimization ideas.
- Tie recommendations back to the repository goal: improve `output_tokens_per_sec` without regressing quality or exceeding memory limits.

## Command Workflow

1. Verify that `xctrace` is available.

```bash
DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer" xctrace version
```

2. Export the trace table of contents.

Use the `trace_toc` tool first. If you need the raw command, this is the equivalent:

```bash
DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer" xctrace export --input "path/to/trace.trace" --toc
```

3. Inspect the exported TOC and identify the run number, process, and relevant data tables.

4. Export only the entities needed for analysis.

```bash
DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer" xctrace export --input "path/to/trace.trace" --xpath '/trace-toc/run[@number="1"]/data/table[@schema="..."]'
```

5. Repeat exports until you can answer the analysis questions below with evidence.

Do not guess table schemas in advance. Use the TOC first, then export the specific runs and tables that the trace actually contains.

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
- run number
- process name
- exported tables used as evidence

Example:

```text
Inference window:
start = 12.184 s
end   = 12.226 s
total = 42.0 ms
```

Use signposts, process names, command-buffer labels, or repeated inference markers to isolate the window. If there are multiple plausible windows, state which one you chose and why.

### 2. Identify the Critical Path

Determine whether total runtime is dominated by GPU execution, CPU-side submission/setup delays, or both.

Measure or estimate from exported tables:

- total GPU execution time in the inference window
- GPU idle gaps between kernel groups
- CPU activity around command submission
- whether kernels are tightly packed or separated by waits

Rules of thumb:

- If GPU activity occupies most of the window, classify it as GPU-bound.
- If kernels are short and separated by meaningful idle gaps while CPU stays active, classify it as CPU-bound or submission-bound.
- If both are substantial, classify it as mixed.

Compute when the data is available:

- sum of GPU kernel durations
- union of GPU busy intervals when possible
- total gap time between kernels or encoder groups
- CPU-side setup or submission time where exposed

Example:

```text
Total inference time: 42 ms
Total GPU active time: 38 ms
GPU idle gaps inside window: 4 ms
Conclusion: primarily GPU-bound, with some CPU submission overhead
```

### 3. Measure Time Attribution

Group events by operation or kernel family and sum total duration.

For GPU-bound cases:

- group GPU events by kernel or shader name
- sort by total time descending
- report count, total duration, average duration, and percent of inference time

For CPU-bound cases:

- group CPU-side functions, stacks, categories, or submission markers
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

This comparison is often easier after exporting structured tables than by visual inspection alone.

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

These patterns show up as durations, counts, gap structure, transfer markers, and event-sequence structure.

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

Focus the recommendations on code changes that can plausibly be made in `generate.py` or the generation flow around it. Do not stop at trace description; convert the trace evidence into the smallest high-leverage code changes.

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
- Trace file: ___
- Exported tables used: ___
- Run number: ___
- Inference start: ___
- Inference end: ___
- Total inference time: ___ ms
- Hardware: ___
- Trace source: `xctrace export`

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

- Use the `trace_toc` tool for TOC export, then `bash` + `xctrace` for targeted table exports.
- Prefer the `DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"` prefix in direct `xctrace` commands unless the environment is already configured.
- Start with the TOC, then export only the relevant tables.
- Be quantitative and specific.
- Prefer direct measurements over guesses.
- If `xctrace` is unavailable or the input is not a `.trace` file, state the blocker immediately.
- Tie every recommendation back to evidence from the trace.
