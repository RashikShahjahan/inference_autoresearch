# autoresearch

This repo is an inference-oriented take on the minimal `karpathy/autoresearch` setup, and full credit to Andrej Karpathy for the original style and core concept. Instead of optimizing a training loop, the agent optimizes generation throughput for `bn -> en` translation with `mlx-community/translategemma-4b-it-4bit` on a fixed subset of `google/wmt24pp`. The point is that you are not manually doing inference research in the usual way. You set up the harness, define the contract, point the agent at `program.md`, and let it iterate on `generate.py`. The benchmark is strict: candidate output token ids must exactly match the frozen reference outputs, and every candidate must stay under a configurable peak Metal memory ceiling.

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** - fixed benchmark contract, setup, reference generation, logging, and incumbent management. Not modified during research.
- **`generate.py`** - the single file the agent edits. It contains the mutable generate-path implementation.
- **`program.md`** - baseline instructions for the autonomous coding agent.

By design, benchmarking runs against a fixed translation fixture and uses **`output_tokens_per_sec`** as the objective metric - higher is better. Correctness is exact-match on output token ids, and **`max_peak_metal_mb`** in `config.json` is a hard memory constraint. So the search space is narrow by construction: same task, same fixtures, same outputs, same hardware budget, one mutable file.

## Quick start

**Requirements:** Apple Silicon / MLX-capable environment, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Set the memory ceiling in config.json, then initialize the sandbox
uv run prepare.py setup

# 4. Run a quick benchmark trial
uv run generate.py --description "trial change"

# 5. Run the full benchmark; only full runs can promote a new incumbent
uv run generate.py --full --description "candidate change"
```

If the above commands all work, the sandbox is initialized and you can start autonomous experimentation.

## Running the agent

Point your coding agent at this repo and have it read `program.md`, `prepare.py`, `generate.py`, and `config.json`.

For example:

```text
Hi, have a look at program.md and let's kick off a new experiment. Start with setup and then iterate on generate.py.
```

The `program.md` file is the lightweight research policy for the agent: what it may change, what the benchmark enforces, and how incumbent promotion works.

The intended workflow is simple: the agent makes a small change, runs the benchmark, checks whether throughput improved under the real contract, and either keeps the change or discards it. In the morning, you inspect `results.tsv` and the incumbent snapshot instead of trying to remember which decode-loop tweak helped.

## Project structure

```text
prepare.py      - fixed benchmark setup and utilities
generate.py     - mutable generate-path candidate
program.md      - agent instructions
config.json     - benchmark contract and dataset selection
pyproject.toml  - dependencies
state/          - incumbent snapshot and frozen references
runs/           - per-run JSON artifacts
results.tsv     - append-only run log
```

## Design choices

- **Single file to modify.** The agent only edits `generate.py`, which keeps the search space constrained and diffs easy to review.
- **Fixed correctness contract.** Candidate token ids must match the frozen reference outputs exactly, so throughput wins are real implementation wins rather than behavioral drift.
- **Real benchmark gate.** Quick runs are for iteration, but only full runs can promote a new incumbent.
- **Inference-specific.** The system optimizes generate-path throughput under a Metal memory ceiling, not model quality or translation style.

## Platform support

This repo is built around MLX inference and currently assumes an Apple Silicon environment with Metal memory reporting available. The default benchmark target is `mlx-community/translategemma-4b-it-4bit`, and the harness enforces a peak Metal memory ceiling via `config.json`.

If you adapt this to other hardware or runtimes, you will likely need a different benchmarking harness and platform-specific memory accounting. The overall pattern should transfer cleanly, but the current repo is intentionally narrow and opinionated.

## Config

`config.json` defines the benchmark contract. The main fields are:

- `model`
- `source_lang`
- `target_lang`
- `dataset_repo`
- `dataset_file`
- `dataset_text_field`
- `dataset_fixture_limit`
- `dataset_skip_bad_source`
- `max_new_tokens`
- `warmup_runs`
- `quick_repeats`
- `full_repeats`
- `max_peak_metal_mb`
- `min_keep_gain_percent`
- `tie_memory_delta_mb`
- `quick_fixture_ids`

`max_peak_metal_mb` must be set to a positive value before `uv run prepare.py setup` or benchmark runs will execute.

## License

MIT
