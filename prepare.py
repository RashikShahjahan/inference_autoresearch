from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATASET_CACHE_DIR = ROOT / ".cache" / "huggingface"
CONFIG_PATH = ROOT / "config.json"
GENERATE_PATH = ROOT / "generate.py"
RESULTS_PATH = ROOT / "results.tsv"
STATE_DIR = ROOT / "state"
INCUMBENT_PATH = STATE_DIR / "best_generate.py"


@dataclass(frozen=True)
class Config:
    model: str
    source_lang: str
    target_lang: str
    dataset_repo: str
    dataset_file: str
    dataset_source_field: str
    dataset_reference_field: str
    dataset_fixture_limit: int | None
    dataset_skip_bad_source: bool
    max_new_tokens: int
    warmup_runs: int
    repeats: int
    max_peak_metal_mb: float | None


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    source_text: str
    reference_text: str
    max_tokens: int | None = None


def load_config() -> Config:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    max_peak_metal_mb = payload.get("max_peak_metal_mb")
    if max_peak_metal_mb is not None:
        max_peak_metal_mb = float(max_peak_metal_mb)
    dataset_source_field = str(payload["dataset_source_field"])
    dataset_reference_field = payload.get("dataset_reference_field")
    if dataset_reference_field is None:
        if dataset_source_field == "source":
            dataset_reference_field = "target"
        elif dataset_source_field == "target":
            dataset_reference_field = "source"
        else:
            raise ValueError(
                "config.json must set dataset_reference_field when dataset_source_field is not 'source' or 'target'"
            )
    return Config(
        model=str(payload["model"]),
        source_lang=str(payload["source_lang"]),
        target_lang=str(payload["target_lang"]),
        dataset_repo=str(payload["dataset_repo"]),
        dataset_file=str(payload["dataset_file"]),
        dataset_source_field=dataset_source_field,
        dataset_reference_field=str(dataset_reference_field),
        dataset_fixture_limit=(
            int(payload["dataset_fixture_limit"])
            if payload.get("dataset_fixture_limit") is not None
            else None
        ),
        dataset_skip_bad_source=bool(payload["dataset_skip_bad_source"]),
        max_new_tokens=int(payload["max_new_tokens"]),
        warmup_runs=int(payload["warmup_runs"]),
        repeats=int(payload["repeats"]),
        max_peak_metal_mb=max_peak_metal_mb,
    )


def require_memory_limit(config: Config) -> None:
    if config.max_peak_metal_mb is None or config.max_peak_metal_mb <= 0:
        raise ValueError("config.json must set max_peak_metal_mb to a positive value")


def dataset_fixture_id(payload: dict, line_number: int) -> str:
    lp = str(payload.get("lp") or "row")
    segment_id = payload.get("segment_id")
    if segment_id is None:
        return f"{lp}-{line_number:04d}"
    return f"{lp}-{int(segment_id):04d}"


def load_fixtures() -> list[Fixture]:
    from huggingface_hub import hf_hub_download

    config = load_config()
    dataset_path = Path(
        hf_hub_download(
            repo_id=config.dataset_repo,
            filename=config.dataset_file,
            repo_type="dataset",
            cache_dir=DATASET_CACHE_DIR,
        )
    )
    fixtures: list[Fixture] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if config.dataset_skip_bad_source and payload.get("is_bad_source"):
            continue
        source_text = str(payload.get(config.dataset_source_field, "")).strip()
        reference_text = str(payload.get(config.dataset_reference_field, "")).strip()
        if not source_text or not reference_text:
            continue
        fixture_id = dataset_fixture_id(payload, line_number)
        if fixture_id in seen_ids:
            raise ValueError(
                f"Duplicate fixture id at line {line_number}: {fixture_id}"
            )
        seen_ids.add(fixture_id)
        fixtures.append(
            Fixture(
                fixture_id=fixture_id,
                source_text=source_text,
                reference_text=reference_text,
            )
        )
        if (
            config.dataset_fixture_limit is not None
            and len(fixtures) >= config.dataset_fixture_limit
        ):
            break
    if not fixtures:
        raise ValueError(
            f"No usable fixtures found in {config.dataset_repo}/{config.dataset_file}"
        )
    return fixtures


def load_model_and_tokenizer(config: Config):
    from mlx_lm import load

    model, tokenizer = load(config.model)
    tokenizer.add_eos_token("<end_of_turn>")
    return model, tokenizer


def build_prompt(tokenizer, config: Config, source_text: str):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": config.source_lang,
                    "target_lang_code": config.target_lang,
                    "text": source_text.strip(),
                }
            ],
        }
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def max_tokens_for_fixture(config: Config, fixture: Fixture) -> int:
    return (
        fixture.max_tokens if fixture.max_tokens is not None else config.max_new_tokens
    )


def profile_batch_generate_metal(
    batch_generate_fn,
    model,
    tokenizer,
    prompts,
    *,
    max_tokens: int | list[int] = 128,
    trace_path: str | Path = "state/batch_generate_profile.gputrace",
    warmup: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Capture a single representative ``batch_generate`` call as a Metal trace."""
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("Metal profiling requires an Apple Silicon / Metal device")

    trace_path = Path(trace_path)
    if trace_path.suffix != ".gputrace":
        raise ValueError("trace_path must end with .gputrace")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        if trace_path.is_dir():
            shutil.rmtree(trace_path)
        else:
            trace_path.unlink()

    synchronize = getattr(mx, "synchronize", None)
    reset_peak_memory = getattr(mx, "reset_peak_memory", None)
    get_peak_memory = getattr(mx, "get_peak_memory", None)

    if warmup:
        batch_generate_fn(
            model,
            tokenizer,
            prompts,
            max_tokens=max_tokens,
            **kwargs,
        )
        if callable(synchronize):
            synchronize()

    if callable(reset_peak_memory):
        reset_peak_memory()
    else:
        mx.metal.reset_peak_memory()
    if callable(synchronize):
        synchronize()

    started = time.perf_counter()
    capture_started = False
    try:
        mx.metal.start_capture(str(trace_path))
        capture_started = True
        response = batch_generate_fn(
            model,
            tokenizer,
            prompts,
            max_tokens=max_tokens,
            **kwargs,
        )
        if callable(synchronize):
            synchronize()
    except RuntimeError as exc:
        if "Capture layer is not inserted" in str(exc):
            raise RuntimeError(
                "Metal capture requires launching the process with MTL_CAPTURE_ENABLED=1"
            ) from exc
        raise
    finally:
        if capture_started:
            mx.metal.stop_capture()

    elapsed = time.perf_counter() - started
    peak_memory_bytes = (
        int(get_peak_memory())
        if callable(get_peak_memory)
        else int(mx.metal.get_peak_memory())
    )

    return {
        "trace_path": str(trace_path),
        "prompt_count": len(prompts),
        "output_tokens": sum(len(token_ids) for token_ids in response.token_ids),
        "elapsed_seconds": round(elapsed, 4),
        "peak_metal_mb": round(peak_memory_bytes / 1024 / 1024, 1),
        "warmup": warmup,
    }


def benchmark_generate_fn(generate_fn, model, tokenizer, config: Config, fixtures):
    import mlx.core as mx
    from sacrebleu.metrics import CHRF

    repeats = config.repeats
    prompts_by_max_tokens: dict[int, list[list[int]]] = {}
    fixtures_by_max_tokens: dict[int, list[Fixture]] = {}
    warmup_prompt_batch: list[list[int]] | None = None
    warmup_max_tokens: int | None = None
    chrf = CHRF()
    fixture_count = 0
    for fixture in fixtures:
        fixture_max_tokens = max_tokens_for_fixture(config, fixture)
        prompt = build_prompt(tokenizer, config, fixture.source_text)
        if warmup_prompt_batch is None:
            warmup_prompt_batch = [prompt]
            warmup_max_tokens = fixture_max_tokens
        prompts_by_max_tokens.setdefault(fixture_max_tokens, []).append(prompt)
        fixtures_by_max_tokens.setdefault(fixture_max_tokens, []).append(fixture)
        fixture_count += 1

    for _ in range(config.warmup_runs):
        if warmup_prompt_batch is not None and warmup_max_tokens is not None:
            generate_fn(
                model,
                tokenizer,
                warmup_prompt_batch,
                max_tokens=warmup_max_tokens,
            )

    reset = getattr(mx, "reset_peak_memory", None)
    if callable(reset):
        reset()
    else:
        mx.metal.reset_peak_memory()
    synchronize = getattr(mx, "synchronize", None)
    if callable(synchronize):
        synchronize()

    started = time.perf_counter()
    total_output_tokens = 0
    hypotheses: list[str] = []
    references: list[str] = []
    for repeat_index in range(repeats):
        for fixture_max_tokens, prompt_batch in prompts_by_max_tokens.items():
            batch_results = generate_fn(
                model,
                tokenizer,
                prompt_batch,
                max_tokens=fixture_max_tokens,
            )
            if len(batch_results) != len(prompt_batch):
                raise RuntimeError(
                    "Candidate returned the wrong number of outputs for the prompt batch"
                )
            if repeat_index == 0:
                for fixture, result in zip(
                    fixtures_by_max_tokens[fixture_max_tokens], batch_results
                ):
                    hypotheses.append(str(result["text"]).strip())
                    references.append(fixture.reference_text)
            for result in batch_results:
                total_output_tokens += len(result["token_ids"])

    if callable(synchronize):
        synchronize()
    elapsed = time.perf_counter() - started
    get_peak_memory = getattr(mx, "get_peak_memory", None)
    peak_memory_bytes = (
        int(get_peak_memory())
        if callable(get_peak_memory)
        else int(mx.metal.get_peak_memory())
    )
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
        "mode": "full",
        "fixture_count": fixture_count,
        "repeats": repeats,
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


def ensure_results_header():
    if RESULTS_PATH.exists():
        return
    RESULTS_PATH.write_text(
        "run_id\tmode\tcandidate_hash\tincumbent_hash\tcandidate_tps\tincumbent_tps\tpeak_metal_mb\tstatus\tdescription\n",
        encoding="utf-8",
    )


def append_results_row(row: list[str]):
    ensure_results_header()
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(row)


def promote_candidate():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(GENERATE_PATH, INCUMBENT_PATH)


def compare_candidate(config: Config, fixtures, description: str, generate_fn):
    if not INCUMBENT_PATH.exists():
        raise ValueError("Incumbent snapshot missing. Run `uv run prepare.py` first.")

    candidate_file_hash = candidate_hash(GENERATE_PATH)
    incumbent_file_hash = candidate_hash(INCUMBENT_PATH)
    model, tokenizer = load_model_and_tokenizer(config)
    candidate_metrics = benchmark_generate_fn(
        generate_fn, model, tokenizer, config, fixtures
    )

    if candidate_file_hash == incumbent_file_hash:
        incumbent_metrics = dict(candidate_metrics)
    else:
        incumbent_module = load_module_from_path(
            INCUMBENT_PATH, f"incumbent_{time.time_ns()}"
        )
        incumbent_metrics = benchmark_generate_fn(
            incumbent_module.generate_text, model, tokenizer, config, fixtures
        )

    if not candidate_metrics["ok"]:
        status = "discard"
        decision_reason = candidate_metrics["failure_reason"]
    elif not incumbent_metrics["ok"]:
        raise RuntimeError("Incumbent benchmark failed; rerun `uv run prepare.py`.")
    elif candidate_file_hash == incumbent_file_hash:
        status = "incumbent"
        decision_reason = "same_as_incumbent"
    elif float(candidate_metrics["chrf_score"]) < float(
        incumbent_metrics["chrf_score"]
    ):
        status = "discard"
        decision_reason = "quality_regression"
    elif float(candidate_metrics["output_tokens_per_sec"]) > float(
        incumbent_metrics["output_tokens_per_sec"]
    ):
        promote_candidate()
        status = "promoted"
        decision_reason = "throughput_win"
    else:
        status = "discard"
        decision_reason = "no_throughput_win"

    run_identifier = time.strftime("%Y%m%d-%H%M%S")
    append_results_row(
        [
            run_identifier,
            "full",
            candidate_file_hash,
            incumbent_file_hash,
            f"{float(candidate_metrics.get('output_tokens_per_sec', 0.0)):.4f}",
            f"{float(incumbent_metrics.get('output_tokens_per_sec', 0.0)):.4f}",
            f"{float(candidate_metrics.get('peak_metal_mb', 0.0)):.1f}",
            status,
            description,
        ]
    )

    return {
        "run_id": run_identifier,
        "mode": "full",
        "description": description,
        "candidate": candidate_metrics,
        "incumbent": incumbent_metrics,
        "status": status,
        "decision_reason": decision_reason,
    }


def main() -> int:
    config = load_config()
    require_memory_limit(config)
    load_fixtures()
    promote_candidate()
    ensure_results_header()
    print(
        json.dumps(
            {
                "status": "initialized",
                "incumbent": str(INCUMBENT_PATH),
                "results": str(RESULTS_PATH),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
