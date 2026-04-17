# autoresearch

This is an experiment to have the LLM do its own inference research.

## Setup

1. Set `max_peak_metal_mb` in `config.json` to a positive value.
2. Read `README.md`, `prepare.py`, `generate.py`, and `config.json`.
3. Run `uv run prepare.py` once to seed the incumbent and run log.
4. Confirm `state/best_generate.py` and `results.tsv` exist.

## Experimentation

The repo is intentionally narrow.

**What you CAN do:**
- Modify `generate.py`.

**What you CANNOT do:**
- Change the dataset selection or quick fixture ids in `config.json` during normal experimentation.
- Modify the benchmark logic in `prepare.py` during normal experimentation.
- Add new external dependencies or rely on libraries that are not already available in this repo.

**Goal:** maximize `output_tokens_per_sec` while staying at or below `max_peak_metal_mb`.

## Loop

1. Edit `generate.py` with one concrete idea.
2. Run `uv run generate.py --description "describe the change"`.
3. If the quick run still looks good, run `uv run generate.py --full --description "describe the change"`.
4. Trust the benchmark result. Quick wins are logged as `trial`. Full wins are logged as `promoted` and automatically replace `state/best_generate.py`.
5. If the result is `discard`, revert or overwrite the change before the next idea.
6. Repeat.

## Output format

After a benchmark run, the CLI prints a JSON summary like this:

```json
{
  "run_id": "20260417-120000",
  "mode": "quick",
  "description": "prefill tweak",
  "candidate": {
    "ok": true,
    "mode": "quick",
    "fixture_count": 2,
    "repeats": 2,
    "elapsed_seconds": 1.2345,
    "output_tokens": 987,
    "output_tokens_per_sec": 799.5132,
    "peak_metal_mb": 12345.6,
    "max_peak_metal_mb": 13000.0,
    "failure_reason": null
  },
  "incumbent": {
    "ok": true,
    "mode": "quick",
    "fixture_count": 2,
    "repeats": 2,
    "elapsed_seconds": 1.252,
    "output_tokens": 987,
    "output_tokens_per_sec": 788.3387,
    "peak_metal_mb": 12380.4,
    "max_peak_metal_mb": 13000.0,
    "failure_reason": null
  },
  "status": "trial",
  "decision_reason": "quick_throughput_win"
}
```

If memory exceeds the ceiling, the candidate result returns `failure_reason: "memory_limit_exceeded"`.
