# inference_assistant

This repo is a compact inference benchmarking setup for optimizing inference throughput on Apple Silicon.

## How it works

The repo intentionally keeps the workflow small:

- `prepare.py` owns the fixed benchmark contract, setup, incumbent snapshot, and result logging.
- `generate.py` is the only file you tune. It delegates comparison to the fixed benchmark.
- `config.json` defines the model, dataset slice, memory ceiling, and repeat counts.

Performance is measured as `output_tokens_per_sec`. `max_peak_metal_mb` is a hard limit.

## Quick start

**Requirements:** Apple Silicon / MLX-capable environment, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Set the memory ceiling in config.json, then seed the benchmark state
uv run prepare.py

# 4. Run a quick benchmark
uv run generate.py --description "trial change"

# 5. Run the full benchmark
uv run generate.py --full --description "candidate change"
```

## Project structure

```text
prepare.py      - fixed benchmark setup, incumbent snapshot, and logging
generate.py     - mutable batched generate-path candidate
program.md      - research assistant instructions
config.json     - benchmark contract and dataset selection
pyproject.toml  - dependencies
state/          - incumbent snapshot
results.tsv     - append-only run log
```

## Workflow

1. Run `uv run prepare.py` once after changing the benchmark contract.
2. Edit `generate.py`.
3. Run `uv run generate.py --description "..."` for the quick fixture set.
4. If the quick run looks good, run `uv run generate.py --full --description "..."`.
5. Full runs automatically promote the candidate to `state/best_generate.py` when throughput improves and memory stays within the ceiling.
6. Inspect `results.tsv` for experiment history.

## Config

`config.json` defines the benchmark contract:

- `model`
- `source_lang`
- `target_lang`
- `dataset_repo`
- `dataset_file`
- `dataset_source_field`
- `dataset_fixture_limit`
- `dataset_skip_bad_source`
- `max_new_tokens`
- `warmup_runs`
- `quick_repeats`
- `full_repeats`
- `max_peak_metal_mb`
- `quick_fixture_ids`

`max_peak_metal_mb` must be set to a positive value before `uv run prepare.py` or benchmark runs will execute.

## License

MIT
