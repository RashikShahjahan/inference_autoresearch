from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


@contextlib.contextmanager
def _buffer_stderr_on_success():
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        try:
            yield
        except Exception:
            sys.stderr.write(buffer.getvalue())
            raise


def _tool_result(payload: dict) -> dict:
    return {
        "output": json.dumps(payload, indent=2),
        "metadata": payload,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize or validate benchmark state"
    )
    parser.add_argument(
        "--overwrite-incumbent",
        action="store_true",
        help="Replace state/best_generate.py with the current generate.py even if it already exists",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    with _buffer_stderr_on_success():
        from prepare import (
            GENERATE_PATH,
            INCUMBENT_PATH,
            RESULTS_PATH,
            RESULTS_HEADER,
            STATE_DIR,
            ensure_results_header,
            load_config,
            load_fixtures,
            require_memory_limit,
            require_supported_model_config,
            SUPPORTED_MODEL_FAMILY,
        )

        config = load_config()
        require_supported_model_config(config)
        require_memory_limit(config)
        fixtures = load_fixtures(config.dataset_fixture_limit)

        STATE_DIR.mkdir(parents=True, exist_ok=True)
        incumbent_existed_before = INCUMBENT_PATH.exists()
        promoted = False
        if args.overwrite_incumbent or not incumbent_existed_before:
            shutil.copy2(GENERATE_PATH, INCUMBENT_PATH)
            promoted = True

        ensure_results_header()
        header = RESULTS_PATH.read_text(encoding="utf-8").splitlines()[0]
        result = {
            "status": "initialized" if promoted else "already_initialized",
            "promoted_current_generate": promoted,
            "incumbent_existed_before": incumbent_existed_before,
            "incumbent": str(INCUMBENT_PATH),
            "results": str(RESULTS_PATH),
            "results_header_ok": f"{header}\n" == RESULTS_HEADER,
            "fixture_count": len(fixtures),
            "config": {
                "model": config.model,
                "supported_model_family": SUPPORTED_MODEL_FAMILY,
                "source_lang": config.source_lang,
                "target_lang": config.target_lang,
                "dataset_repo": config.dataset_repo,
                "dataset_fixture_limit": config.dataset_fixture_limit,
                "max_new_tokens": config.max_new_tokens,
                "max_peak_metal_mb": config.max_peak_metal_mb,
            },
        }
        print(json.dumps(_tool_result(result), indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
