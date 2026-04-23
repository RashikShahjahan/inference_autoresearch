from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TARGET_SCRIPT = Path(__file__).with_name("capture_gpu_trace_target.py")
XCTRACE_TEMPLATE = "Metal System Trace"
MEASUREMENT_DELAY_SECONDS = 2.0
XCODE_DEVELOPER_DIR = Path("/Applications/Xcode.app/Contents/Developer")
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
        description="Capture a representative Instruments Metal System Trace for generate.py"
    )
    parser.add_argument(
        "--trace-path",
        default="state/batch_generate_profile.trace",
        help="Write an Instruments .trace for one representative batch_generate call",
    )
    parser.add_argument(
        "--fixture-count",
        type=int,
        default=1,
        help="Number of fixtures to profile as one representative batch",
    )
    return parser


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def remove_existing_output(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def build_xctrace_env() -> dict[str, str]:
    env = dict(os.environ)
    if "DEVELOPER_DIR" in env:
        return env
    if XCODE_DEVELOPER_DIR.is_dir():
        env["DEVELOPER_DIR"] = str(XCODE_DEVELOPER_DIR)
    return env


def ensure_xctrace_available(env: dict[str, str]) -> None:
    try:
        subprocess.run(
            ["xctrace", "version"],
            check=True,
            capture_output=True,
            text=True,
            cwd=WORKSPACE_ROOT,
            env=env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "xctrace was not found. Install full Xcode and make sure it is the active developer directory."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(
            "xctrace is unavailable. Install full Xcode and set it as the active developer directory."
            + (f"\n{detail}" if detail else "")
        ) from exc


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    with _buffer_stderr_on_success():
        fixture_count = args.fixture_count
        if fixture_count <= 0:
            raise ValueError("fixture count must be positive")

        trace_path = resolve_repo_path(args.trace_path)
        if trace_path.suffix != ".trace":
            raise ValueError("trace_path must end with .trace")

        xctrace_env = build_xctrace_env()
        ensure_xctrace_available(xctrace_env)
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=trace_path.parent) as temp_dir:
            temp_trace_path = Path(temp_dir) / "captured.trace"
            result_path = Path(temp_dir) / "capture_result.json"
            command = [
                "xctrace",
                "record",
                "--template",
                XCTRACE_TEMPLATE,
                "--output",
                str(temp_trace_path),
                "--no-prompt",
                "--launch",
                "--",
                sys.executable,
                str(TARGET_SCRIPT),
                "--fixture-count",
                str(fixture_count),
                "--result-path",
                str(result_path),
                "--measurement-delay-seconds",
                str(MEASUREMENT_DELAY_SECONDS),
            ]

            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=WORKSPACE_ROOT,
                    env=xctrace_env,
                )
            except subprocess.CalledProcessError as exc:
                detail = "\n".join(
                    part.strip()
                    for part in (exc.stderr, exc.stdout)
                    if part and part.strip()
                )
                raise RuntimeError(
                    "xctrace record failed"
                    + (f"\n{detail}" if detail else "")
                ) from exc

            if not result_path.exists():
                raise RuntimeError(
                    "xctrace completed but the traced target did not write its summary"
                )
            if not temp_trace_path.exists():
                raise RuntimeError("xctrace completed but did not write the .trace output")

            remove_existing_output(trace_path)
            shutil.move(str(temp_trace_path), str(trace_path))

            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["trace_path"] = str(trace_path)
            result["trace_format"] = "trace"
            result["xctrace_template"] = XCTRACE_TEMPLATE
            result["developer_dir"] = xctrace_env.get("DEVELOPER_DIR")

        print(json.dumps(_tool_result(result), indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
