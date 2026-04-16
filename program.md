## Goal

Optimize `runtime.py` to maximize `output_tokens_per_sec` for `bn -> en` translation under a hard `max_peak_metal_mb` ceiling.

## Allowed changes

- Edit only `runtime.py`

## Forbidden changes

- Do not edit `baseline.py`
- Do not edit `harness.py`
- Do not edit `config.json`
- Do not edit `fixtures/benchmark.jsonl`
- Do not edit files outside `generate_autoresearch/`

## Benchmark rules

- Correctness is strict: candidate output token ids must exactly match the frozen baseline
- Peak Metal memory must stay at or below `max_peak_metal_mb`
- Quick runs are for iteration only
- Only `eval --full` can promote a new incumbent

## Workflow

1. Start from the current incumbent with `uv run python generate_autoresearch/run.py reset`
2. Make one small change to `runtime.py`
3. Run a quick check:

```bash
uv run python generate_autoresearch/run.py eval --description "describe the change"
```

4. If the quick run looks promising, run the full benchmark:

```bash
uv run python generate_autoresearch/run.py eval --full --description "describe the change"
```

5. Trust the harness decision

## Good search directions

- prefill chunk sizing
- one-shot vs chunked prefill
- `mx.eval()` cadence
- `mx.clear_cache()` policy
- token collection strategy
- incremental detokenization vs final decode
- local reference caching in the decode loop
- stop-token checks

## Out of scope

- batching
- worker scheduling
- database or app changes
- model swaps
- prompt-template changes
- speculative decoding
