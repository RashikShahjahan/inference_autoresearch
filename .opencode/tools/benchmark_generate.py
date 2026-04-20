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
    parser = argparse.ArgumentParser(description="Benchmark the current generate.py")
    parser.add_argument(
        "--description",
        default="manual run",
        help="Short description of the current experiment",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    with _buffer_stderr_on_success():
        from generate import generate_text
        from inference_workflow import compare_candidate
        from prepare import load_config, load_fixtures, require_memory_limit

        config = load_config()
        fixtures = load_fixtures()
        require_memory_limit(config)

        result = compare_candidate(config, fixtures, args.description, generate_text)
        print(json.dumps(_tool_result(result), indent=2))
        return 0 if result["status"] in {"promoted", "incumbent"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
