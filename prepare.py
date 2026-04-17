from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


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
    dataset_fixture_limit: int | None
    dataset_skip_bad_source: bool
    max_new_tokens: int
    warmup_runs: int
    quick_repeats: int
    full_repeats: int
    max_peak_metal_mb: float | None
    quick_fixture_ids: tuple[str, ...]


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    source_text: str
    max_tokens: int | None = None


def load_config() -> Config:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    max_peak_metal_mb = payload.get("max_peak_metal_mb")
    if max_peak_metal_mb is not None:
        max_peak_metal_mb = float(max_peak_metal_mb)
    return Config(
        model=str(payload["model"]),
        source_lang=str(payload["source_lang"]),
        target_lang=str(payload["target_lang"]),
        dataset_repo=str(payload["dataset_repo"]),
        dataset_file=str(payload["dataset_file"]),
        dataset_source_field=str(payload["dataset_source_field"]),
        dataset_fixture_limit=(
            int(payload["dataset_fixture_limit"])
            if payload.get("dataset_fixture_limit") is not None
            else None
        ),
        dataset_skip_bad_source=bool(payload["dataset_skip_bad_source"]),
        max_new_tokens=int(payload["max_new_tokens"]),
        warmup_runs=int(payload["warmup_runs"]),
        quick_repeats=int(payload["quick_repeats"]),
        full_repeats=int(payload["full_repeats"]),
        max_peak_metal_mb=max_peak_metal_mb,
        quick_fixture_ids=tuple(str(item) for item in payload["quick_fixture_ids"]),
    )


def require_memory_limit(config: Config):
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
        if not source_text:
            continue
        fixture_id = dataset_fixture_id(payload, line_number)
        if fixture_id in seen_ids:
            raise ValueError(
                f"Duplicate fixture id at line {line_number}: {fixture_id}"
            )
        seen_ids.add(fixture_id)
        fixtures.append(Fixture(fixture_id=fixture_id, source_text=source_text))
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


def split_fixtures(
    config: Config, fixtures: list[Fixture]
) -> tuple[list[Fixture], list[Fixture]]:
    fixture_by_id = {fixture.fixture_id: fixture for fixture in fixtures}
    quick: list[Fixture] = []
    for fixture_id in config.quick_fixture_ids:
        fixture = fixture_by_id.get(fixture_id)
        if fixture is None:
            raise ValueError(f"Quick fixture id not found: {fixture_id}")
        quick.append(fixture)
    return quick, fixtures


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


def benchmark_generate_fn(
    generate_fn, model, tokenizer, mode: str, config: Config, fixtures
):
    import mlx.core as mx

    repeats = config.quick_repeats if mode == "quick" else config.full_repeats
    prompts_by_max_tokens: dict[int, list[list[int]]] = {}
    fixture_count = 0
    for fixture in fixtures:
        fixture_max_tokens = max_tokens_for_fixture(config, fixture)
        prompts_by_max_tokens.setdefault(fixture_max_tokens, []).append(
            build_prompt(tokenizer, config, fixture.source_text)
        )
        fixture_count += 1

    for _ in range(config.warmup_runs):
        for fixture_max_tokens, prompt_batch in prompts_by_max_tokens.items():
            generate_fn(model, tokenizer, prompt_batch, max_tokens=fixture_max_tokens)

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
    for _ in range(repeats):
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

    return {
        "ok": within_memory_limit,
        "mode": mode,
        "fixture_count": fixture_count,
        "repeats": repeats,
        "elapsed_seconds": round(elapsed, 4),
        "output_tokens": total_output_tokens,
        "output_tokens_per_sec": round(output_tokens_per_sec, 4),
        "peak_metal_mb": peak_metal_mb,
        "max_peak_metal_mb": float(config.max_peak_metal_mb),
        "failure_reason": None if within_memory_limit else "memory_limit_exceeded",
    }


def load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
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


def compare_candidate(
    config: Config, mode: str, fixtures, description: str, generate_fn
):
    if not INCUMBENT_PATH.exists():
        raise ValueError("Incumbent snapshot missing. Run `uv run prepare.py` first.")

    candidate_file_hash = candidate_hash(GENERATE_PATH)
    incumbent_file_hash = candidate_hash(INCUMBENT_PATH)
    model, tokenizer = load_model_and_tokenizer(config)
    candidate_metrics = benchmark_generate_fn(
        generate_fn, model, tokenizer, mode, config, fixtures
    )

    if candidate_file_hash == incumbent_file_hash:
        incumbent_metrics = dict(candidate_metrics)
    else:
        incumbent_module = load_module_from_path(
            INCUMBENT_PATH, f"incumbent_{time.time_ns()}"
        )
        incumbent_metrics = benchmark_generate_fn(
            incumbent_module.generate_text, model, tokenizer, mode, config, fixtures
        )

    if not candidate_metrics["ok"]:
        status = "discard"
        decision_reason = candidate_metrics["failure_reason"]
    elif not incumbent_metrics["ok"]:
        raise RuntimeError("Incumbent benchmark failed; rerun `uv run prepare.py`.")
    elif candidate_file_hash == incumbent_file_hash:
        status = "incumbent"
        decision_reason = "same_as_incumbent"
    elif float(candidate_metrics["output_tokens_per_sec"]) > float(
        incumbent_metrics["output_tokens_per_sec"]
    ):
        if mode == "full":
            promote_candidate()
            status = "promoted"
            decision_reason = "throughput_win"
        else:
            status = "trial"
            decision_reason = "quick_throughput_win"
    else:
        status = "discard"
        decision_reason = "no_throughput_win"

    run_identifier = time.strftime("%Y%m%d-%H%M%S")
    append_results_row(
        [
            run_identifier,
            mode,
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
        "mode": mode,
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
