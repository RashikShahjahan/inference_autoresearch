# autoresearch

This repo is a small sandbox for autonomous inference-path research on MLX.

The idea: give an agent a fixed translation benchmark and a single mutable `generate.py`, let it try generate-path changes, measure throughput, and keep only the changes that beat the incumbent under the real contract.

Scope:
- language pair: `bn -> en`
- default model: `mlx-community/translategemma-4b-it-4bit`
- objective: maximize `output_tokens_per_sec`
- hard constraint: stay under a configurable peak Metal memory ceiling
- correctness: candidate output token ids must match the frozen reference exactly

The layout intentionally mirrors the minimal shape of `karpathy/autoresearch`, but for inference instead of training.

## How it works

The repo is deliberately small and only really has three top-level files that matter:

- `prepare.py` — fixed benchmark contract, setup, reference generation, and state management. Do not modify during research.
- `generate.py` — the single file the agent edits. It contains the mutable generate-path implementation.
- `program.md` — instructions for the autonomous coding agent.

Supporting files:

- `config.json` — benchmark contract and memory ceiling
- `fixtures/benchmark.jsonl` — checked-in Bangla prompts
- `state/` — incumbent snapshot and frozen reference outputs
- `runs/` — per-run JSON artifacts
- `results.tsv` — append-only run log

## Quick start

Set the peak Metal memory ceiling in `config.json`, then initialize the sandbox:

```bash
uv run prepare.py setup
```

Run a quick benchmark on the configured subset:

```bash
uv run generate.py --description "trial change"
```

Run the full benchmark. Only full runs can promote a new incumbent:

```bash
uv run generate.py --full --description "candidate change"
```

Restore `generate.py` from the current incumbent snapshot:

```bash
uv run prepare.py reset
```

Show the current incumbent and recent results:

```bash
uv run prepare.py status
```

## Project structure

```text
prepare.py      fixed benchmark setup and utilities
generate.py     mutable generate-path candidate
program.md      agent instructions
config.json     benchmark contract
fixtures/       benchmark prompts
```

## Design choices

- Single mutable file. The agent only edits `generate.py`.
- Fixed correctness contract. Candidate token ids must match the frozen reference outputs exactly.
- Fixed harness. Benchmarking, setup, logging, and incumbent promotion live in `prepare.py`.
- Inference-specific. This repo optimizes generate-path throughput, not model quality.

## Config

`config.json` controls the benchmark contract. The main fields are:

- `model`
- `source_lang`
- `target_lang`
- `max_new_tokens`
- `max_peak_metal_mb`

`max_peak_metal_mb` must be set before `setup` or benchmark runs will execute.
