# Generate Autoresearch

This directory is a self-contained sandbox for translation generate-path research on MLX.

Scope:
- language pair: `bn -> en`
- default model: `mlx-community/translategemma-4b-it-4bit`
- objective: maximize `output_tokens_per_sec`
- hard constraint: stay under a configurable peak Metal memory ceiling
- correctness: candidate output tokens must match the frozen baseline exactly

Rules:
- Do not import code from `src/`
- Do not edit files outside `generate_autoresearch/`
- During research, only `runtime.py` should change

## Files

- `baseline.py`: frozen reference generate implementation
- `runtime.py`: mutable candidate implementation
- `harness.py`: fixed benchmark, scoring, and MLX utilities
- `run.py`: CLI for setup/eval/reset/status
- `program.md`: lightweight instructions for an autonomous coding agent
- `fixtures/benchmark.jsonl`: checked-in Bangla benchmark prompts

## Commands

Set up the sandbox first:

```bash
uv run python generate_autoresearch/run.py setup
```

Run a quick comparison on the configured subset:

```bash
uv run python generate_autoresearch/run.py eval --description "trial change"
```

Run the full benchmark. Only full runs can promote a new incumbent:

```bash
uv run python generate_autoresearch/run.py eval --full --description "candidate change"
```

Restore `runtime.py` from the current incumbent snapshot:

```bash
uv run python generate_autoresearch/run.py reset
```

Show the current incumbent and recent results:

```bash
uv run python generate_autoresearch/run.py status
```

## Config

`config.json` controls the benchmark contract. `max_peak_metal_mb` is required and must be set before `setup` or `eval` will run.
