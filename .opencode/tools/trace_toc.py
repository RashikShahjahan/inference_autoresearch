from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
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
    parser = argparse.ArgumentParser(description="Export the xctrace table of contents for a .trace file")
    parser.add_argument(
        "--trace-path",
        default="state/batch_generate_profile.trace",
        help="Path to a .trace document, relative to repo root or absolute",
    )
    return parser


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


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
    args = build_parser().parse_args()

    with _buffer_stderr_on_success():
        trace_path = resolve_repo_path(args.trace_path)
        if not trace_path.exists():
            raise FileNotFoundError(f"Trace path does not exist: {trace_path}")
        if trace_path.suffix != ".trace":
            raise ValueError("trace_path must end with .trace")

        env = build_xctrace_env()
        ensure_xctrace_available(env)

        try:
            completed = subprocess.run(
                ["xctrace", "export", "--input", str(trace_path), "--toc"],
                check=True,
                capture_output=True,
                text=True,
                cwd=WORKSPACE_ROOT,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            detail = "\n".join(
                part.strip() for part in (exc.stderr, exc.stdout) if part and part.strip()
            )
            raise RuntimeError("xctrace export --toc failed" + (f"\n{detail}" if detail else "")) from exc

        payload = {
            "trace_path": str(trace_path),
            "developer_dir": env.get("DEVELOPER_DIR"),
            "toc_xml": completed.stdout,
        }
        print(json.dumps(_tool_result(payload), indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
