from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET_CACHE_DIR = ROOT / ".cache" / "huggingface"
CONFIG_PATH = ROOT / "config.json"
GENERATE_PATH = ROOT / "generate.py"
RESULTS_PATH = ROOT / "results.tsv"
STATE_DIR = ROOT / "state"
INCUMBENT_PATH = STATE_DIR / "best_generate.py"
RESULTS_HEADER = (
    "run_id\tcandidate_hash\tincumbent_hash\tmlx_lm_tps\tcandidate_tps\t"
    "incumbent_tps\tpeak_metal_mb\tstatus\tdescription\n"
)
SUPPORTED_MODEL_FAMILY = "translategemma"


@dataclass(frozen=True)
class Config:
    model: str
    source_lang: str
    target_lang: str
    dataset_repo: str
    dataset_source_field: str
    dataset_reference_field: str
    dataset_fixture_limit: int | None
    max_new_tokens: int
    max_peak_metal_mb: float | None


@dataclass(frozen=True)
class Fixture:
    source_text: str
    reference_text: str


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
        dataset_source_field=str(payload["dataset_source_field"]),
        dataset_reference_field=str(payload["dataset_reference_field"]),
        dataset_fixture_limit=(
            int(payload["dataset_fixture_limit"])
            if payload.get("dataset_fixture_limit") is not None
            else 1
        ),
        max_new_tokens=int(payload["max_new_tokens"]),
        max_peak_metal_mb=max_peak_metal_mb,
    )


def require_memory_limit(config: Config) -> None:
    if config.max_peak_metal_mb is None or config.max_peak_metal_mb <= 0:
        raise ValueError("config.json must set max_peak_metal_mb to a positive value")


def require_supported_model_config(config: Config) -> None:
    if SUPPORTED_MODEL_FAMILY not in config.model.lower():
        raise ValueError(
            "This benchmark harness is specialized for TranslatedGemma models; "
            f"config.json has model={config.model!r}"
        )


def require_supported_model_runtime(model) -> None:
    language_model = getattr(model, "language_model", None)
    text_model = getattr(language_model, "model", None)
    if (
        getattr(model, "model_type", None) != "gemma3"
        or text_model is None
        or not (
            hasattr(language_model, "lm_head")
            or getattr(language_model, "tie_word_embeddings", False)
        )
    ):
        raise ValueError(
            "Loaded model does not expose the TranslatedGemma/Gemma3 text path "
            "required by generate.py"
        )


def load_fixtures(fixture_limit: int) -> list[Fixture]:
    from huggingface_hub import hf_hub_download, list_repo_files

    config = load_config()
    dataset_files = sorted(
        path for path in list_repo_files(config.dataset_repo, repo_type="dataset") if path.endswith(".jsonl")
    )

    dataset_path = Path(
        hf_hub_download(
            repo_id=config.dataset_repo,
            filename=dataset_files[0],
            repo_type="dataset",
            cache_dir=DATASET_CACHE_DIR,
        )
    )
    fixtures: list[Fixture] = []

    for line_number, raw_line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line_number > fixture_limit:
            break

        payload = json.loads(raw_line)
        source_text = str(payload[config.dataset_source_field]).strip()
        reference_text = str(payload[config.dataset_reference_field]).strip()

        fixtures.append(
            Fixture(
                source_text=source_text,
                reference_text=reference_text,
            )
        )
    if not fixtures:
        raise ValueError(f"No fixtures found in {config.dataset_repo}")
    return fixtures


def load_model_and_tokenizer(config: Config):
    from mlx_lm import load

    require_supported_model_config(config)
    model, tokenizer = load(config.model)
    require_supported_model_runtime(model)
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


def ensure_results_header():
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(RESULTS_HEADER, encoding="utf-8")
        return

    lines = RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    if not lines:
        RESULTS_PATH.write_text(RESULTS_HEADER, encoding="utf-8")
        return
    if f"{lines[0]}\n" == RESULTS_HEADER:
        return

    raise ValueError(f"Unexpected results.tsv header: {lines[0]!r}")


def promote_candidate():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(GENERATE_PATH, INCUMBENT_PATH)


def main() -> int:
    config = load_config()
    require_supported_model_config(config)
    require_memory_limit(config)
    load_fixtures(config.dataset_fixture_limit)
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
