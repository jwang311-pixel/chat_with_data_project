"""
clean_results.py

删除 results/ 目录下所有 CSV 文件中，指定模型的所有行。
运行方式：python clean_results.py
"""

import csv
from pathlib import Path

RESULTS_DIR = Path("results")

# 要清理的模型，这三个模型的所有数据都会被删掉
MODELS_TO_CLEAN = [
    "nvidia/nemotron-3-super-120b-a12b:free",
]


def clean_csv(path: Path) -> None:
    if not path.exists():
        print(f"[SKIP] {path.name} not found")
        return

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    original_count = len(rows)

    # 只保留不在清理列表里的模型的行
    kept_rows = [
        row for row in rows
        if row.get("model_id") not in MODELS_TO_CLEAN
    ]

    removed_count = original_count - len(kept_rows)

    if removed_count == 0:
        print(f"[OK] {path.name}: no rows to remove")
        return

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"[CLEANED] {path.name}: removed {removed_count} rows, kept {len(kept_rows)} rows")


def main():
    csv_files = sorted(RESULTS_DIR.glob("*_results.csv"))

    if not csv_files:
        print(f"No result CSV files found in {RESULTS_DIR}/")
        return

    print(f"Cleaning all rows for models:")
    for m in MODELS_TO_CLEAN:
        print(f"  - {m}")
    print()

    for csv_file in csv_files:
        clean_csv(csv_file)

    print("\nDone.")


if __name__ == "__main__":
    main()