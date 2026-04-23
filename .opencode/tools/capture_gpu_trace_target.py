from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one warmed batch_generate call inside an xctrace-launched process"
    )
    parser.add_argument(
        "--fixture-count",
        type=int,
        required=True,
        help="Number of fixtures to include in the representative batch",
    )
    parser.add_argument(
        "--result-path",
        required=True,
        help="Write a JSON summary for the outer capture tool",
    )
    parser.add_argument(
        "--measurement-delay-seconds",
        type=float,
        default=2.0,
        help="Delay after warmup to create a clear gap before the measured call",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.fixture_count <= 0:
        raise ValueError("fixture count must be positive")
    if args.measurement_delay_seconds < 0:
        raise ValueError("measurement delay seconds must be non-negative")

    import mlx.core as mx

    from generate import batch_generate
    from prepare import (
        build_prompt,
        load_config,
        load_fixtures,
        load_model_and_tokenizer,
    )

    config = load_config()
    selected_fixtures = load_fixtures(args.fixture_count)
    if len(selected_fixtures) != args.fixture_count:
        raise ValueError(
            f"fixture count {args.fixture_count} exceeds available fixtures {len(selected_fixtures)}"
        )

    model, tokenizer = load_model_and_tokenizer(config)
    prompts = [
        build_prompt(tokenizer, config, fixture.source_text)
        for fixture in selected_fixtures
    ]
    max_tokens = [config.max_new_tokens] * len(selected_fixtures)

    batch_generate(
        model,
        tokenizer,
        prompts,
        max_tokens=max_tokens,
    )
    mx.synchronize()

    mx.metal.reset_peak_memory()
    mx.synchronize()

    if args.measurement_delay_seconds:
        time.sleep(args.measurement_delay_seconds)

    started = time.perf_counter()
    response = batch_generate(
        model,
        tokenizer,
        prompts,
        max_tokens=max_tokens,
    )
    mx.synchronize()

    elapsed = time.perf_counter() - started
    peak_memory_bytes = int(mx.metal.get_peak_memory())
    result = {
        "prompt_count": len(prompts),
        "output_tokens": sum(len(token_ids) for token_ids in response.token_ids),
        "elapsed_seconds": round(elapsed, 4),
        "peak_metal_mb": round(peak_memory_bytes / 1024 / 1024, 1),
        "warmup": True,
        "measurement_delay_seconds": round(args.measurement_delay_seconds, 3),
    }

    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
