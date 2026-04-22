from __future__ import annotations

import argparse
import contextlib
import io
import json
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
        description="Capture a representative Metal trace for generate.py"
    )
    parser.add_argument(
        "--metal-profile-path",
        default="state/batch_generate_profile.gputrace",
        help="Write a Metal GPU trace for one representative batch_generate call",
    )
    parser.add_argument(
        "--metal-profile-fixture-count",
        type=int,
        default=1,
        help="Number of fixtures to profile as one representative batch",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    with _buffer_stderr_on_success():
        from generate import batch_generate
        from inference_workflow import profile_batch_generate_metal
        from prepare import (
            build_prompt,
            load_config,
            load_fixtures,
            load_model_and_tokenizer,
        )

        config = load_config()
        fixtures = load_fixtures()

        fixture_count = args.metal_profile_fixture_count
        if fixture_count <= 0:
            raise ValueError("metal profile fixture count must be positive")

        selected_fixtures = fixtures[:fixture_count]
        if len(selected_fixtures) != fixture_count:
            raise ValueError(
                f"metal profile fixture count {fixture_count} exceeds available fixtures {len(fixtures)}"
            )

        model, tokenizer = load_model_and_tokenizer(config)
        prompts = [
            build_prompt(tokenizer, config, fixture.source_text)
            for fixture in selected_fixtures
        ]
        max_tokens = [config.max_new_tokens] * len(selected_fixtures)
        result = profile_batch_generate_metal(
            batch_generate,
            model,
            tokenizer,
            prompts,
            max_tokens=max_tokens,
            trace_path=args.metal_profile_path,
        )
        print(json.dumps(_tool_result(result), indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
