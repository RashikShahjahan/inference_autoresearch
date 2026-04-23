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
I follow a fixed 5-step reasoning process:

1. Establish the baseline inference window.
2. Identify the critical path.
3. Measure time attribution.
4. Record trace-observed inefficiencies.
5. Prioritize and hypothesize optimizations.

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

2. Export the trace table of contents using the `trace_toc` tool.

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

Classify the window using the measured distribution of GPU active time, GPU idle gaps, and CPU submission activity. State the evidence for the classification instead of relying on fixed thresholds alone.

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

### 4. Record Trace-Observed Inefficiencies

Only include issues that are directly supported by exported trace data. Do not rely on a fixed anti-pattern catalog or numeric thresholds.

Look for and record any of the following when they are present in the trace:

- repeated GPU idle gaps inside the inference window
- repeated CPU submission or synchronization waits
- transfer or blit events that take measurable time
- extra kernel launches or repeated event sequences within the inference window
- repeated short kernel groups that coincide with idle gaps or host-side waits

For each issue you report, include:

- the specific trace evidence
- where it appears in the inference window
- why it matters for end-to-end inference time

If the trace does not support a particular issue category, say so rather than inferring it.

### 5. Prioritize and Hypothesize

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
- repeated matmul launch groups within the same inference window
- repeated 0.5-1.0 ms gaps between kernel groups
Expected improvement:
- reduced end-to-end inference time if launch fragmentation and synchronization overhead are lowered
Proposed fix:
- replace manual attention sequence with fused MLX attention path
- reduce eager sync points
- batch operations where possible
```

Focus the recommendations on code changes that can plausibly be made in `generate.py` or the generation flow around it. Do not stop at trace description; convert the trace evidence into the smallest high-leverage code changes.


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

## 4. Trace-Observed Issues
- GPU idle gaps: ___
- Transfer or blit events: ___
- CPU submission or synchronization waits: ___
- Extra launches or repeated event sequences: ___
- Dominant issue: ___

## 5. Prioritized Actions
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
