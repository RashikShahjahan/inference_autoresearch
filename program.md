# autoresearch

This is an experiment to have the LLM do its own inference research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on the memory ceiling**: set `max_peak_metal_mb` in `config.json`. The repo refuses to run setup or benchmarks until this is a positive value.
2. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed benchmark contract, setup, reference generation, and state management. Do not modify during research.
   - `generate.py` — the file you modify.
   - `config.json` — benchmark contract, especially `model`, `source_lang`, `target_lang`, `max_new_tokens`, and the peak Metal memory limit.
3. **Initialize the sandbox**: run `uv run prepare.py setup`.
4. **Verify state exists**: `setup` should create the reference outputs, the incumbent snapshot, the best-metrics file, and `results.tsv`.
5. **Confirm and go**: Confirm setup looks good. The setup run establishes the baseline and syncs `generate.py` to the incumbent.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs through the fixed benchmark contract. Iterate with quick runs, and only use full runs to advance the incumbent.

**What you CAN do:**
- Modify `generate.py` — this is the only file you edit during research.
- Change generate-path implementation details such as prefill chunk sizing, one-shot vs chunked prefill, `mx.eval()` cadence, `mx.clear_cache()` policy, token collection strategy, detokenization timing, local reference caching in the decode loop, and stop-token checks.

**What you CANNOT do:**
- Modify `prepare.py`.
- Modify `fixtures/benchmark.jsonl`.
- Change the correctness contract: candidate output token ids must exactly match the frozen reference outputs.
- Use batching, worker scheduling changes, model swaps, prompt-template changes, or speculative decoding.

**The goal is simple: maximize `output_tokens_per_sec` while staying at or below `max_peak_metal_mb`.** Correctness is strict, and the peak Metal memory ceiling is a hard constraint.

**Promotion rules**: Quick runs are for iteration only. Only `uv run generate.py --full --description "..."` can promote a new incumbent.

**Simplicity criterion**: All else being equal, simpler is better. A tiny throughput win that adds awkward complexity is usually not worth it. Equal throughput with meaningfully lower memory is a win. Equal or better results from less code is an especially good outcome.

**The first run**: Your first run should always be `uv run prepare.py setup`.

## Output format

After a benchmark run, the CLI prints a JSON summary like this:

```json
{
  "run_id": "20260416-123456",
  "mode": "quick",
  "description": "adjust prefill chunk size",
  "candidate": {
    "ok": true,
    "candidate_hash": "abc123def456",
    "mode": "quick",
    "fixture_count": 3,
    "repeats": 2,
    "elapsed_seconds": 1.2345,
    "output_tokens": 987,
    "output_tokens_per_sec": 799.5132,
    "peak_metal_mb": 12345.6
  },
  "incumbent": {
    "ok": true,
    "candidate_hash": "def456abc123",
    "mode": "quick",
    "fixture_count": 3,
    "repeats": 2,
    "elapsed_seconds": 1.2520,
    "output_tokens": 987,
    "output_tokens_per_sec": 788.3387,
    "peak_metal_mb": 12380.4
  },
  "status": "trial",
  "decision_reason": "quick_win_1.42_percent"
}
```

The exact numbers will vary by machine and by change. The fields that matter most are `candidate.output_tokens_per_sec`, `candidate.peak_metal_mb`, `status`, and `decision_reason`.

## Logging results

When an experiment finishes, the benchmark appends a row to `results.tsv` automatically. Do not hand-maintain it during normal runs.

The TSV has a header row and 7 columns:

```text
run_id	mode	candidate_hash	output_tokens_per_sec	peak_metal_mb	status	description
```

1. run identifier, formatted like `YYYYMMDD-HHMMSS`
2. benchmark mode: `quick` or `full`
3. content hash of the candidate `generate.py`
4. candidate throughput in output tokens per second
5. candidate peak Metal memory in MB
6. benchmark decision: `trial`, `promoted`, or `discard`
7. short description of what the experiment tried

## The experiment loop

The incumbent lives in `state/best_generate.py`, and `uv run prepare.py reset` restores `generate.py` from that snapshot.

LOOP FOREVER:

1. Look at the current state with `uv run prepare.py status`.
2. Reset `generate.py` from the incumbent with `uv run prepare.py reset`.
3. Tune `generate.py` with one concrete idea.
4. Run a quick benchmark: `uv run generate.py --description "describe the change"`.
5. If the quick run looks promising, run the full benchmark: `uv run generate.py --full --description "describe the change"`.
6. Trust the benchmark decision. If the result is `promoted`, the incumbent advanced automatically. If the result is `discard`, reset and move on. If the result is `trial`, you only have quick evidence so far.
7. Inspect `results.tsv` or `runs/<run_id>/result.json` if you need the recorded metrics after a run.
8. Repeat with the next idea.

The idea is that you are an autonomous researcher making small, testable changes to `generate.py`. Keep changes that beat the incumbent under the real benchmark contract, and discard changes that do not.

**Failures**: If a run crashes because of an obvious bug, fix it and rerun. If the idea itself is fundamentally bad or violates the correctness or memory contract, discard it and move on.

**NEVER STOP**: Once the experiment loop has begun, do not pause to ask the human if you should continue. Keep iterating until you are manually stopped.
