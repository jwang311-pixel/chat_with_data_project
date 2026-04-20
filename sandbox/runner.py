from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.utils.csv_utils import read_csv_flexible  # noqa: E402


def _safe_builtins():
    return {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }


def serialize_result(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)


def main() -> int:
    payload = json.loads(sys.stdin.read())
    csv_path = payload["csv_path"]
    python_code = payload["python_code"]

    df = read_csv_flexible(csv_path)
    safe_globals = {
        "__builtins__": _safe_builtins(),
        "pd": pd,
        "np": np,
        "df": df,
        "result": None,
    }
    safe_locals = {}
    stdout_buffer = io.StringIO()

    result_value = None  # 提前初始化，防止 exec 前报错时 result_value 未定义

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(python_code, safe_globals, safe_locals)
        result_value = safe_locals.get("result", safe_globals.get("result"))
        output = {
            "stdout": stdout_buffer.getvalue(),
            "result": serialize_result(result_value),  # 统一使用 "result"
            "error": None,
        }
        print(json.dumps(output, ensure_ascii=False))
        return 0
    except Exception as exc:
        output = {
            "stdout": stdout_buffer.getvalue(),
            "result": serialize_result(result_value),  # 统一使用 "result"
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(output, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())