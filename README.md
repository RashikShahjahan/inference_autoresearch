# inference_assistant

This repo is a compact inference benchmarking setup for optimizing inference throughput on Apple Silicon.


## Quick start

**Requirements:** Apple Silicon / MLX-capable environment, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Set the memory ceiling in config.json, then seed the benchmark state
uv run prepare.py
```

Then run the `benchmark_generate` OpenCode tool with `description="candidate change"`.


## Trace analysis

Use the `capture_gpu_trace` OpenCode tool to record a representative Instruments Metal trace at
`state/batch_generate_profile.trace`.

Inspect the trace contents with the `trace_toc` OpenCode tool, then use `xctrace export` from `bash`
for targeted table exports. If the active developer directory is still Command Line Tools, prefix
direct `xctrace` commands with:

```bash
DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
```


## Config

`config.json` defines the benchmark contract:

- `model`
- `source_lang`
- `target_lang`
- `dataset_repo`
- `dataset_source_field`
- `dataset_reference_field`
- `dataset_fixture_limit`
- `max_new_tokens`
- `max_peak_metal_mb`

`dataset_repo` is expected to contain exactly one `.jsonl` file.

`dataset_fixture_limit` caps how many dataset rows are loaded.

`max_peak_metal_mb` must be set to a positive value before `uv run prepare.py` or benchmark runs will execute.

`dataset_reference_field` is the reference translation field used for quality scoring on the same fixtures.

Every benchmark uses one warmup run followed by one measured pass for `mlx_lm.batch_generate`, the current candidate, and the incumbent snapshot.

`results.tsv` includes `mlx_lm_tps` alongside candidate and incumbent throughput so you can track progress against both baselines over time.

## License

MIT
