from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass

from app.core.config import SANDBOX_DIR


@dataclass
class SandboxResult:
    stdout: str
    result_repr: str | None
    error: str | None
    raw: dict


class SandboxService:
    def __init__(self) -> None:
        self.runner_path = SANDBOX_DIR / "runner.py"

    def execute(self, *, csv_path: str, python_code: str, timeout_seconds: int = 20) -> SandboxResult:
        payload = json.dumps({"csv_path": csv_path, "python_code": python_code})
        proc = subprocess.run(
            [sys.executable, str(self.runner_path)],
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        if proc.returncode != 0:
            return SandboxResult(
                stdout=proc.stdout,
                result_repr=None,
                error=proc.stderr or proc.stdout or "Sandbox execution failed",
                raw={"returncode": proc.returncode},
            )

        raw = json.loads(proc.stdout)
        return SandboxResult(
            stdout=raw.get("stdout", ""),
            result_repr=raw.get("result"),  # 与 runner.py 统一，读取 "result" 字段
            error=raw.get("error"),
            raw=raw,
        )