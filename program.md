# inference_assistant

This repository is set up for the `inference_assistant` research assistant workflow for inference benchmarking.

## Setup

To start or resume a run, work through this checklist:

1. Ensure `config.json` sets `max_peak_metal_mb` to a positive value.
2. Read the in-scope files for full context:
   - `README.md` - repository context and workflow.
   - `prepare.py` - fixed benchmark contract, fixture loading, memory checks, incumbent promotion, and results logging. Do not modify during normal experimentation.
   - `generate.py` - the only file you tune.
   - `config.json` - fixed benchmark configuration for the current run.
3. Run `uv run prepare.py` once after changing the benchmark contract or when initializing a fresh workspace.
4. Verify that `state/best_generate.py` and `results.tsv` exist.
5. Confirm the benchmark is ready, then summarize the current state for the user.

## Working style

Each experiment benchmarks `generate.py` against the incumbent snapshot in `state/best_generate.py`.



**What you CAN do:**
- Modify `generate.py`; Make sure that the improvements cannot simply be achieved by changing function parameters and require a change in the generation logic. Changes in any layer of the implementation are fair game.

**What you CANNOT do during normal experimentation unless the user asks:**
- Modify `prepare.py` or the benchmark logic it defines.
- Change the dataset selection or memory ceiling in `config.json`.
- Add new dependencies or rely on packages that are not already present in `pyproject.toml`.

**The goal is simple: maximize `output_tokens_per_sec` while staying at or below `max_peak_metal_mb`.**

Memory is a hard constraint. If a candidate exceeds `max_peak_metal_mb`, it is automatically discarded.

**Simplicity criterion:** all else equal, prefer the simpler change. A small throughput gain is not worth a pile of brittle complexity. If a simpler implementation matches or slightly improves throughput while respecting memory, that is a strong result.

**The first run:** `uv run prepare.py`. 

## Output format

After a benchmark run, the CLI prints a JSON summary like this:

```json
{
  "run_id": "20260417-120000",
  "mode": "full",
  "description": "prefill tweak",
  "candidate": {
    "ok": true,
    "mode": "full",
    "fixture_count": 32,
    "repeats": 3,
    "elapsed_seconds": 1.2345,
    "output_tokens": 987,
    "output_tokens_per_sec": 799.5132,
    "quality_metric": "chrf",
    "quality_fixture_count": 2,
    "chrf_score": 54.321,
    "peak_metal_mb": 12345.6,
    "max_peak_metal_mb": 13000.0,
    "failure_reason": null
  },
  "incumbent": {
    "ok": true,
    "mode": "full",
    "fixture_count": 32,
    "repeats": 3,
    "elapsed_seconds": 1.252,
    "output_tokens": 987,
    "output_tokens_per_sec": 788.3387,
    "quality_metric": "chrf",
    "quality_fixture_count": 2,
    "chrf_score": 53.998,
    "peak_metal_mb": 12380.4,
    "max_peak_metal_mb": 13000.0,
    "failure_reason": null
  },
  "status": "promoted",
  "decision_reason": "throughput_win"
}
```

The key fields are:

- `candidate.output_tokens_per_sec`: the metric to maximize.
- `candidate.chrf_score`: translation quality against the reference field on the same fixtures, computed with `sacrebleu` corpus `chrF`.
- `candidate.peak_metal_mb`: must stay within the configured ceiling.
- `status`: one of `incumbent`, `promoted`, or `discard`.
- `decision_reason`: explains why the candidate was kept or rejected.

Candidates are discarded when `chrf_score` is lower than the incumbent, even if throughput is higher.

If memory exceeds the ceiling, the candidate returns `failure_reason: "memory_limit_exceeded"` and the run is discarded.

## Logging results

Every benchmark appends a row to `results.tsv` automatically. The file is tab-separated and has this header:

```tsv
run_id	mode	candidate_hash	incumbent_hash	candidate_tps	incumbent_tps	peak_metal_mb	status	description
```

The benchmark owns this log. Do not hand-edit it during normal experimentation.

## Suggested workflow

1. Inspect the current candidate in `generate.py` and the benchmark contract in `README.md`, `prepare.py`, and `config.json` when needed.
2. Explain the current bottleneck, risk, or opportunity.
3. Propose one concrete change and the expected tradeoff.
4. Implement the change in `generate.py`.
5. Run the benchmark: `uv run generate.py --description "describe the change"`.
6. Summarize the outcome:
   - `promoted` means the full candidate beat the incumbent and `state/best_generate.py` was updated automatically.
   - `discard` means the candidate lost on throughput or violated the memory ceiling.
   - `incumbent` means the file is effectively identical to the current best snapshot.
7. Ask whether the user wants to continue with another idea.
