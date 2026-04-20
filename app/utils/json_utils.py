from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def make_json_safe(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, pd.DataFrame):
        cleaned = obj.replace([np.inf, -np.inf], np.nan).where(pd.notna(obj), None)
        return cleaned.to_dict(orient="records")

    if isinstance(obj, pd.Series):
        cleaned = obj.replace([np.inf, -np.inf], np.nan).where(pd.notna(obj), None)
        return cleaned.to_list()

    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]

    return obj
    