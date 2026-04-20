from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd


def read_csv_flexible(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf-8-sig", errors="replace")
    lines = raw_text.splitlines()
    if not lines:
        return pd.DataFrame()

    # Some sample CSVs start with a title row before the real header.
    first_line = lines[0].strip()
    if first_line and "," not in first_line and len(lines) > 1:
        lines = lines[1:]

    normalized = "\n".join(lines)
    return pd.read_csv(StringIO(normalized))


def dataframe_preview(df: pd.DataFrame, rows: int = 3) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return df.head(rows).to_dict(orient="records")


def infer_dataframe_summary(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = [c for c in df.columns if c not in numeric_cols]
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "text_columns": text_cols,
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "preview": dataframe_preview(df, rows=3),
    }
