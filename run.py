from __future__ import annotations

import argparse
import json

from harness import (
    compare_candidate,
    incumbent_path,
    initialize_state,
    load_best_metrics,
    load_config,
    load_fixtures,
    recent_results,
    require_memory_limit,
    reset_runtime_from_incumbent,
    split_fixtures,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the generate autoresearch sandbox"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "setup", help="Initialize reference outputs and incumbent state"
    )

    eval_parser = subparsers.add_parser("eval", help="Benchmark the current runtime")
    eval_parser.add_argument(
        "--full", action="store_true", help="Use the full fixture set"
    )
    eval_parser.add_argument(
        "--description",
        default="manual run",
        help="Short description of the candidate change",
    )

    subparsers.add_parser(
        "reset", help="Restore runtime.py from the incumbent snapshot"
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
    quick_fixtures, full_fixtures = split_fixtures(config, fixtures)

    if args.command in {"setup", "eval"}:
        require_memory_limit(config)

    if args.command == "setup":
        baseline_metrics = initialize_state(config, full_fixtures)
        print(
            json.dumps(
                {"status": "initialized", "baseline": baseline_metrics}, indent=2
            )
        )
        return 0

    if args.command == "eval":
        mode = "full" if args.full else "quick"
        selected_fixtures = full_fixtures if args.full else quick_fixtures
        result = compare_candidate(config, mode, selected_fixtures, args.description)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "reset":
        try:
            reset_runtime_from_incumbent()
            runtime_source = str(incumbent_path())
        except ValueError as exc:
            print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
            return 1
        print(
            json.dumps({"status": "reset", "runtime_source": runtime_source}, indent=2)
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
