from __future__ import annotations

import argparse
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
RESULTS_PATH = ROOT / "results.tsv"
RUNS_DIR = ROOT / "runs"
STATE_DIR = ROOT / "state"
REFERENCE_OUTPUTS_PATH = STATE_DIR / "reference_outputs.json"
BEST_GENERATE_PATH = STATE_DIR / "best_generate.py"
BEST_METRICS_PATH = STATE_DIR / "best_metrics.json"
GENERATE_PATH = ROOT / "generate.py"


@dataclass(frozen=True)
class Config:
    model: str
    source_lang: str
    target_lang: str
    dataset_repo: str
    dataset_file: str
    dataset_text_field: str
    dataset_fixture_limit: int | None
    dataset_skip_bad_source: bool
    max_new_tokens: int
    warmup_runs: int
    quick_repeats: int
    full_repeats: int
    max_peak_metal_mb: float | None
    min_keep_gain_percent: float
    tie_memory_delta_mb: float
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
        dataset_text_field=str(payload["dataset_text_field"]),
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
        min_keep_gain_percent=float(payload["min_keep_gain_percent"]),
        tie_memory_delta_mb=float(payload["tie_memory_delta_mb"]),
        quick_fixture_ids=tuple(str(item) for item in payload["quick_fixture_ids"]),
    )


def require_memory_limit(config: Config):
    if config.max_peak_metal_mb is None or config.max_peak_metal_mb <= 0:
        raise ValueError("config.json must set max_peak_metal_mb to a positive value")


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
        source_text = str(payload.get(config.dataset_text_field, "")).strip()
        if not source_text:
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


def dataset_fixture_id(payload: dict, line_number: int) -> str:
    lp = str(payload.get("lp") or "row")
    segment_id = payload.get("segment_id")
    if segment_id is None:
        return f"{lp}-{line_number:04d}"
    return f"{lp}-{int(segment_id):04d}"


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


def sync():
    import mlx.core as mx

    synchronize = getattr(mx, "synchronize", None)
    if callable(synchronize):
        synchronize()


def reset_peak_memory():
    import mlx.core as mx

    reset = getattr(mx, "reset_peak_memory", None)
    if callable(reset):
        reset()
        return
    mx.metal.reset_peak_memory()


def get_peak_memory_bytes() -> int:
    import mlx.core as mx

    getter = getattr(mx, "get_peak_memory", None)
    if callable(getter):
        return int(getter())
    return int(mx.metal.get_peak_memory())


def bytes_to_mb(value: int) -> float:
    return round(value / 1024 / 1024, 1)


def candidate_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:12]


def ensure_results_header():
    if RESULTS_PATH.exists():
        return
    RESULTS_PATH.write_text(
        "run_id\tmode\tcandidate_hash\toutput_tokens_per_sec\tpeak_metal_mb\tstatus\tdescription\n",
        encoding="utf-8",
    )


def append_results_row(row: list[str]):
    ensure_results_header()
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(row)


def load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def generate_references(config: Config, fixtures: list[Fixture], module_path: Path):
    model, tokenizer = load_model_and_tokenizer(config)
    baseline = load_module_from_path(module_path, f"reference_{time.time_ns()}")
    outputs: dict[str, dict] = {}
    for fixture in fixtures:
        prompt_tokens = build_prompt(tokenizer, config, fixture.source_text)
        result = baseline.generate_text(
            model,
            tokenizer,
            prompt_tokens,
            max_tokens=max_tokens_for_fixture(config, fixture),
        )
        outputs[fixture.fixture_id] = {
            "token_ids": [int(token) for token in result["token_ids"]],
        }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_OUTPUTS_PATH.write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_reference_outputs() -> dict[str, dict]:
    if not REFERENCE_OUTPUTS_PATH.exists():
        raise ValueError("Reference outputs missing. Run setup first.")
    return json.loads(REFERENCE_OUTPUTS_PATH.read_text(encoding="utf-8"))


def _run_once(
    module,
    model,
    tokenizer,
    config: Config,
    fixtures: list[Fixture],
    references: dict[str, dict],
):
    total_output_tokens = 0
    for fixture in fixtures:
        prompt_tokens = build_prompt(tokenizer, config, fixture.source_text)
        result = module.generate_text(
            model,
            tokenizer,
            prompt_tokens,
            max_tokens=max_tokens_for_fixture(config, fixture),
        )
        token_ids = [int(token) for token in result["token_ids"]]
        expected = references[fixture.fixture_id]["token_ids"]
        if token_ids != expected:
            return {
                "ok": False,
                "failure_reason": "output_mismatch",
                "fixture_id": fixture.fixture_id,
                "expected_token_count": len(expected),
                "actual_token_count": len(token_ids),
            }
        total_output_tokens += len(token_ids)
    return {
        "ok": True,
        "total_output_tokens": total_output_tokens,
    }


def benchmark_module(
    module_path: Path, mode: str, config: Config, fixtures: list[Fixture]
):
    repeats = config.quick_repeats if mode == "quick" else config.full_repeats
    references = load_reference_outputs()
    model, tokenizer = load_model_and_tokenizer(config)
    module = load_module_from_path(module_path, f"candidate_{time.time_ns()}")

    for _ in range(config.warmup_runs):
        warmup_result = _run_once(
            module, model, tokenizer, config, fixtures, references
        )
        if not warmup_result["ok"]:
            return {
                "ok": False,
                "failure_reason": warmup_result["failure_reason"],
                "fixture_id": warmup_result.get("fixture_id"),
                "module_path": str(module_path),
                "candidate_hash": candidate_hash(module_path),
            }

    reset_peak_memory()
    sync()
    started = time.perf_counter()
    total_output_tokens = 0
    for _ in range(repeats):
        run_result = _run_once(module, model, tokenizer, config, fixtures, references)
        if not run_result["ok"]:
            sync()
            return {
                "ok": False,
                "failure_reason": run_result["failure_reason"],
                "fixture_id": run_result.get("fixture_id"),
                "module_path": str(module_path),
                "candidate_hash": candidate_hash(module_path),
            }
        total_output_tokens += int(run_result["total_output_tokens"])
    sync()
    elapsed = time.perf_counter() - started
    peak_metal_mb = bytes_to_mb(get_peak_memory_bytes())
    output_tokens_per_sec = 0.0 if elapsed <= 0 else total_output_tokens / elapsed

    return {
        "ok": True,
        "module_path": str(module_path),
        "candidate_hash": candidate_hash(module_path),
        "mode": mode,
        "fixture_count": len(fixtures),
        "repeats": repeats,
        "elapsed_seconds": round(elapsed, 4),
        "output_tokens": total_output_tokens,
        "output_tokens_per_sec": round(output_tokens_per_sec, 4),
        "peak_metal_mb": peak_metal_mb,
    }


def run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def save_run_artifact(run_identifier: str, payload: dict):
    output_dir = RUNS_DIR / run_identifier
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def promote_generate(metrics: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(GENERATE_PATH, BEST_GENERATE_PATH)
    BEST_METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_best_metrics() -> dict:
    if not BEST_METRICS_PATH.exists():
        raise ValueError("Best metrics missing. Run setup first.")
    return json.loads(BEST_METRICS_PATH.read_text(encoding="utf-8"))


def incumbent_path() -> Path:
    if not BEST_GENERATE_PATH.exists():
        raise ValueError("Best generate snapshot missing. Run setup first.")
    return BEST_GENERATE_PATH


def reset_generate_from_incumbent():
    shutil.copy2(incumbent_path(), GENERATE_PATH)


def compare_candidate(
    config: Config, mode: str, fixtures: list[Fixture], description: str
):
    candidate_metrics = benchmark_module(GENERATE_PATH, mode, config, fixtures)
    incumbent_metrics = benchmark_module(incumbent_path(), mode, config, fixtures)
    run_identifier = run_id()

    if not candidate_metrics["ok"]:
        status = "discard"
        decision_reason = candidate_metrics["failure_reason"]
    elif candidate_metrics["peak_metal_mb"] > float(config.max_peak_metal_mb):
        status = "discard"
        decision_reason = "memory_limit_exceeded"
    elif not incumbent_metrics["ok"]:
        raise RuntimeError(
            "Incumbent generate.py failed benchmark; reset the sandbox state"
        )
    else:
        incumbent_tps = float(incumbent_metrics["output_tokens_per_sec"])
        candidate_tps = float(candidate_metrics["output_tokens_per_sec"])
        gain_percent = (
            0.0
            if incumbent_tps <= 0
            else ((candidate_tps - incumbent_tps) / incumbent_tps) * 100.0
        )
        memory_delta_mb = float(incumbent_metrics["peak_metal_mb"]) - float(
            candidate_metrics["peak_metal_mb"]
        )

        if gain_percent >= config.min_keep_gain_percent:
            if mode == "full":
                status = "promoted"
                decision_reason = f"throughput_gain_{gain_percent:.2f}_percent"
                promote_generate(candidate_metrics)
            else:
                status = "trial"
                decision_reason = f"quick_win_{gain_percent:.2f}_percent"
        elif (
            abs(gain_percent) < config.min_keep_gain_percent
            and memory_delta_mb >= config.tie_memory_delta_mb
        ):
            if mode == "full":
                status = "promoted"
                decision_reason = f"memory_win_{memory_delta_mb:.1f}_mb"
                promote_generate(candidate_metrics)
            else:
                status = "trial"
                decision_reason = f"quick_memory_win_{memory_delta_mb:.1f}_mb"
        else:
            status = "discard"
            decision_reason = f"no_win_gain_{gain_percent:.2f}_percent_memory_delta_{memory_delta_mb:.1f}_mb"

    artifact = {
        "run_id": run_identifier,
        "mode": mode,
        "description": description,
        "candidate": candidate_metrics,
        "incumbent": incumbent_metrics,
        "status": status,
        "decision_reason": decision_reason,
    }
    save_run_artifact(run_identifier, artifact)

    output_tokens_per_sec = (
        candidate_metrics["output_tokens_per_sec"]
        if candidate_metrics.get("ok")
        else 0.0
    )
    peak_metal_mb = candidate_metrics.get("peak_metal_mb", 0.0)
    append_results_row(
        [
            run_identifier,
            mode,
            candidate_hash(GENERATE_PATH),
            f"{float(output_tokens_per_sec):.4f}",
            f"{float(peak_metal_mb):.1f}",
            status,
            description,
        ]
    )
    return artifact


def initialize_state(config: Config, fixtures: list[Fixture]):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_results_header()
    shutil.copy2(GENERATE_PATH, BEST_GENERATE_PATH)
    generate_references(config, fixtures, BEST_GENERATE_PATH)
    baseline_metrics = benchmark_module(BEST_GENERATE_PATH, "full", config, fixtures)
    if not baseline_metrics["ok"]:
        raise RuntimeError(f"Baseline setup failed: {baseline_metrics}")
    shutil.copy2(BEST_GENERATE_PATH, GENERATE_PATH)
    BEST_METRICS_PATH.write_text(
        json.dumps(baseline_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    append_results_row(
        [
            run_id(),
            "full",
            candidate_hash(BEST_GENERATE_PATH),
            f"{float(baseline_metrics['output_tokens_per_sec']):.4f}",
            f"{float(baseline_metrics['peak_metal_mb']):.1f}",
            "promoted",
            "baseline setup",
        ]
    )
    return baseline_metrics


def recent_results(limit: int = 10) -> list[str]:
    if not RESULTS_PATH.exists():
        return []
    lines = RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    return lines[-limit:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage the generate autoresearch sandbox"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "setup", help="Initialize reference outputs and incumbent state"
    )
    subparsers.add_parser(
        "reset", help="Restore generate.py from the incumbent snapshot"
    )
    subparsers.add_parser(
        "status", help="Show the current incumbent and recent run history"
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config()
    fixtures = load_fixtures()
    _, full_fixtures = split_fixtures(config, fixtures)

    if args.command == "setup":
        require_memory_limit(config)
        baseline_metrics = initialize_state(config, full_fixtures)
        print(
            json.dumps(
                {"status": "initialized", "baseline": baseline_metrics}, indent=2
            )
        )
        return 0

    if args.command == "reset":
        try:
            reset_generate_from_incumbent()
            generate_source = str(incumbent_path())
        except ValueError as exc:
            print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
            return 1
        print(
            json.dumps(
                {"status": "reset", "generate_source": generate_source}, indent=2
            )
        )
        return 0

    if args.command == "status":
        try:
            best_metrics = load_best_metrics()
        except ValueError as exc:
            print(
                json.dumps(
                    {
                        "status": "not_initialized",
                        "message": str(exc),
                        "recent_results": recent_results(),
                    },
                    indent=2,
                )
            )
            return 0
        payload = {
            "best_metrics": best_metrics,
            "recent_results": recent_results(),
        }
        print(json.dumps(payload, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
