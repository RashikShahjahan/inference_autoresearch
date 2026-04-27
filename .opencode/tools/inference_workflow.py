from __future__ import annotations

import csv
import hashlib
import importlib.util
import statistics
import sys
import time
from pathlib import Path

from mlx_lm import batch_generate as mlx_lm_batch_generate

from prepare import (
    GENERATE_PATH,
    INCUMBENT_PATH,
    RESULTS_PATH,
    Config,
    build_prompt,
    ensure_results_header,
    load_model_and_tokenizer,
    promote_candidate,
)


def benchmark_generate_fn(generate_fn, model, tokenizer, config: Config, fixtures):
    import mlx.core as mx
    from sacrebleu.metrics import CHRF

    prompts: list[list[int]] = []
    chrf = CHRF()

    for fixture in fixtures:
        prompts.append(build_prompt(tokenizer, config, fixture.source_text))

    _clear_mlx_state(mx)
    if prompts:
        generate_fn(
            model,
            tokenizer,
            [prompts[0]],
            max_tokens=config.max_new_tokens,
        )

    _clear_mlx_state(mx)
    _reset_peak_memory(mx)

    started = time.perf_counter()
    total_output_tokens = 0
    hypotheses: list[str] = []
    references: list[str] = []

    batch_payload = generate_fn(
        model,
        tokenizer,
        prompts,
        max_tokens=config.max_new_tokens,
    )
    batch_output_tokens: int | None = None
    if isinstance(batch_payload, dict):
        batch_results = batch_payload["results"]
        batch_output_tokens = int(batch_payload.get("output_tokens", 0))
    else:
        batch_results = batch_payload
    for fixture, result in zip(fixtures, batch_results):
        hypotheses.append(str(result["text"]).strip())
        references.append(fixture.reference_text)
    if batch_output_tokens is not None:
        total_output_tokens += batch_output_tokens
    else:
        for result in batch_results:
            total_output_tokens += len(result["token_ids"])

    mx.synchronize()

    elapsed = time.perf_counter() - started
    peak_memory_bytes = int(_get_peak_memory(mx))
    peak_metal_mb = round(peak_memory_bytes / 1024 / 1024, 1)
    output_tokens_per_sec = 0.0 if elapsed <= 0 else total_output_tokens / elapsed
    within_memory_limit = peak_metal_mb <= float(config.max_peak_metal_mb)
    chrf_score = (
        0.0
        if not references
        else float(chrf.corpus_score(hypotheses, [references]).score)
    )

    return {
        "ok": within_memory_limit,
        "fixture_count": len(prompts),
        "elapsed_seconds": round(elapsed, 4),
        "output_tokens": total_output_tokens,
        "output_tokens_per_sec": round(output_tokens_per_sec, 4),
        "quality_metric": "chrf",
        "quality_fixture_count": len(references),
        "chrf_score": round(chrf_score, 4),
        "peak_metal_mb": peak_metal_mb,
        "max_peak_metal_mb": float(config.max_peak_metal_mb),
        "failure_reason": None if within_memory_limit else "memory_limit_exceeded",
    }


def _clear_mlx_state(mx) -> None:
    mx.synchronize()
    mx.clear_cache()
    mx.synchronize()


def _reset_peak_memory(mx) -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:
        mx.metal.reset_peak_memory()


def _get_peak_memory(mx) -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return int(mx.metal.get_peak_memory())


def mlx_lm_generate_text(model, tokenizer, prompt_tokens_batch, *, max_tokens: int):
    response = mlx_lm_batch_generate(
        model,
        tokenizer,
        prompt_tokens_batch,
        max_tokens=max_tokens,
    )
    return {
        "results": [{"text": text, "token_ids": []} for text in response.texts],
        "output_tokens": response.stats.generation_tokens,
    }


def load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def candidate_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def append_results_row(row: list[str]):
    ensure_results_header()
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(row)


def _metric(metrics: dict, key: str) -> float:
    return float(metrics.get(key, 0.0))


def _benchmark_orders(names: list[str]) -> list[list[str]]:
    if len(names) <= 1:
        return [names]
    if len(names) == 2:
        return [names, list(reversed(names))]
    return [names[index:] + names[:index] for index in range(len(names))]


def _aggregate_metrics(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("Cannot aggregate an empty benchmark run list")

    ok = all(run.get("ok", False) for run in runs)
    failure_reason = None
    if not ok:
        failure_reason = next(
            (
                run.get("failure_reason")
                for run in runs
                if run.get("failure_reason") is not None
            ),
            "benchmark_failed",
        )

    tps_samples = [_metric(run, "output_tokens_per_sec") for run in runs]
    chrf_samples = [_metric(run, "chrf_score") for run in runs]
    peak_samples = [_metric(run, "peak_metal_mb") for run in runs]

    return {
        "ok": ok,
        "fixture_count": int(statistics.median(run["fixture_count"] for run in runs)),
        "elapsed_seconds": round(
            statistics.median(_metric(run, "elapsed_seconds") for run in runs), 4
        ),
        "output_tokens": int(
            statistics.median(_metric(run, "output_tokens") for run in runs)
        ),
        "output_tokens_per_sec": round(statistics.median(tps_samples), 4),
        "output_tokens_per_sec_samples": [round(value, 4) for value in tps_samples],
        "quality_metric": runs[0].get("quality_metric", "chrf"),
        "quality_fixture_count": int(
            statistics.median(run["quality_fixture_count"] for run in runs)
        ),
        "chrf_score": round(statistics.median(chrf_samples), 4),
        "chrf_score_samples": [round(value, 4) for value in chrf_samples],
        "peak_metal_mb": round(max(peak_samples), 1),
        "peak_metal_mb_samples": [round(value, 1) for value in peak_samples],
        "max_peak_metal_mb": float(runs[0].get("max_peak_metal_mb", 0.0)),
        "failure_reason": failure_reason,
        "repeats": len(runs),
    }


def _run_balanced_benchmarks(benchmarks: dict[str, object], model, tokenizer, config, fixtures):
    names = list(benchmarks)
    orders = _benchmark_orders(names)
    runs = {name: [] for name in names}

    for order in orders:
        for name in order:
            runs[name].append(
                benchmark_generate_fn(benchmarks[name], model, tokenizer, config, fixtures)
            )

    return {
        "metrics": {name: _aggregate_metrics(name_runs) for name, name_runs in runs.items()},
        "orders": orders,
    }


def compare_candidate(config: Config, fixtures, description: str, generate_fn):
    if not INCUMBENT_PATH.exists():
        raise ValueError("Incumbent snapshot missing. Run `uv run prepare.py` first.")

    candidate_file_hash = candidate_hash(GENERATE_PATH)
    incumbent_file_hash = candidate_hash(INCUMBENT_PATH)
    model, tokenizer = load_model_and_tokenizer(config)

    benchmarks = {
        "candidate": generate_fn,
        "mlx_lm": mlx_lm_generate_text,
    }

    if candidate_file_hash == incumbent_file_hash:
        incumbent_module = None
    else:
        incumbent_module = load_module_from_path(
            INCUMBENT_PATH, f"incumbent_{time.time_ns()}"
        )
        benchmarks["incumbent"] = incumbent_module.generate_text

    benchmark_result = _run_balanced_benchmarks(
        benchmarks, model, tokenizer, config, fixtures
    )
    candidate_metrics = benchmark_result["metrics"]["candidate"]
    mlx_lm_metrics = benchmark_result["metrics"]["mlx_lm"]
    incumbent_metrics = (
        dict(candidate_metrics)
        if incumbent_module is None
        else benchmark_result["metrics"]["incumbent"]
    )

    if not mlx_lm_metrics["ok"]:
        raise RuntimeError(
            "mlx_lm baseline benchmark failed; candidate comparisons require a valid mlx_lm run."
        )

    candidate_tps = _metric(candidate_metrics, "output_tokens_per_sec")
    incumbent_tps = _metric(incumbent_metrics, "output_tokens_per_sec")
    mlx_lm_tps = _metric(mlx_lm_metrics, "output_tokens_per_sec")
    candidate_chrf = _metric(candidate_metrics, "chrf_score")
    incumbent_chrf = _metric(incumbent_metrics, "chrf_score")
    mlx_lm_chrf = _metric(mlx_lm_metrics, "chrf_score")

    if not candidate_metrics["ok"]:
        status = "discard"
        decision_reason = candidate_metrics["failure_reason"]
    elif not incumbent_metrics["ok"]:
        raise RuntimeError("Incumbent benchmark failed; rerun `uv run prepare.py`.")
    elif candidate_file_hash == incumbent_file_hash:
        status = "incumbent"
        decision_reason = (
            "same_as_incumbent"
            if candidate_tps >= mlx_lm_tps and candidate_chrf >= mlx_lm_chrf
            else "same_as_incumbent_below_mlx_lm"
        )
    elif candidate_chrf < incumbent_chrf:
        status = "discard"
        decision_reason = "incumbent_quality_regression"
    elif candidate_chrf < mlx_lm_chrf:
        status = "discard"
        decision_reason = "mlx_lm_quality_regression"
    elif candidate_tps <= incumbent_tps:
        status = "discard"
        decision_reason = "no_incumbent_throughput_win"
    elif candidate_tps <= mlx_lm_tps:
        status = "discard"
        decision_reason = "no_mlx_lm_throughput_win"
    else:
        promote_candidate()
        status = "promoted"
        decision_reason = "throughput_win_vs_incumbent_and_mlx_lm"

    run_identifier = time.strftime("%Y%m%d-%H%M%S")
    append_results_row(
        [
            run_identifier,
            candidate_file_hash,
            incumbent_file_hash,
            f"{mlx_lm_tps:.4f}",
            f"{candidate_tps:.4f}",
            f"{incumbent_tps:.4f}",
            f"{mlx_lm_chrf:.4f}",
            f"{candidate_chrf:.4f}",
            f"{incumbent_chrf:.4f}",
            f"{_metric(mlx_lm_metrics, 'peak_metal_mb'):.1f}",
            f"{_metric(candidate_metrics, 'peak_metal_mb'):.1f}",
            f"{_metric(incumbent_metrics, 'peak_metal_mb'):.1f}",
            status,
            decision_reason,
            description,
        ]
    )

    return {
        "run_id": run_identifier,
        "description": description,
        "mlx_lm": mlx_lm_metrics,
        "candidate": candidate_metrics,
        "incumbent": incumbent_metrics,
        "benchmark_orders": benchmark_result["orders"],
        "status": status,
        "decision_reason": decision_reason,
    }
