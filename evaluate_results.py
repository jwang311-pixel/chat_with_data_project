"""
evaluate_results.py
====================
Evaluation script covering:
  1. Execution Success Rate (by model, by prompt mode)
  2. Answer Accuracy — numeric match + string match (successful rows only)
  3. Prompt Mode Comparison (baseline vs structured vs CoT)
  4. Python Code Efficiency (lines of code, elapsed time by prompt mode)

All output is printed to terminal AND saved to:
  - results/evaluation_report.csv   (detailed per-row results)
  - results/evaluation_summary.json (structured summary)
  - results/evaluation_report.txt   (human-readable full report)

Usage:  python evaluate_results.py
Requires: pip install pandas openpyxl
"""

import re
import io
import json
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────────

RESULT_FILES = {
    "hr_employees":        "results/hr_employees_results.csv",
    "sales_orders":        "results/sales_orders_results.csv",
    "finance_expenses":    "results/finance_expenses_results.csv",
    "marketing_campaigns": "results/marketing_campaigns_results.csv",
    "customer_service":    "results/customer_service_results.csv",
}

GROUND_TRUTH_FILE = "evaluation/questions/ground_truth.xlsx"

GROUND_TRUTH_SHEETS = {
    "hr_employees":        "HR_employees",
    "sales_orders":        "sales_order",
    "finance_expenses":    "finance_analysis",
    "marketing_campaigns": "marketing_campaign",
    "customer_service":    "customer_service",
}

COL_QUESTION_ID  = "question_id"
COL_QUESTION     = "question"
COL_PRED         = "final_answer"
COL_GROUND_TRUTH = "ground_truth"
COL_STATUS       = "status"
COL_PROMPT_MODE  = "prompt_mode"
COL_MODEL        = "model_id"
COL_ELAPSED      = "elapsed_sec"
COL_EXEC_ERROR   = "execution_error"
COL_CODE         = "python_code"

COL_GT_ID     = "question_id"
COL_GT_ANSWER = "answer"

NUMERIC_TOLERANCE = 0.01
SUCCESS_STATUS    = "ok"

OUTPUT_CSV  = "results/evaluation_report.csv"
OUTPUT_JSON = "results/evaluation_summary.json"
OUTPUT_TXT  = "results/evaluation_report.txt"

# ─────────────────────────────────────────────
# Logger — prints to terminal AND captures for txt
# ─────────────────────────────────────────────

class Logger:
    def __init__(self):
        self.lines = []

    def log(self, text=""):
        print(text)
        self.lines.append(text)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(f"💾 TXT report saved: {path}")


logger = Logger()


# ─────────────────────────────────────────────
# 1. Numeric match
# ─────────────────────────────────────────────

def extract_number(text: str):
    if text is None:
        return None
    text = str(text).replace(",", "").strip()
    match = re.search(r"-?\d+\.?\d*", text)
    return float(match.group()) if match else None


def numeric_match(pred: str, gt: str) -> dict:
    pred_num = extract_number(pred)
    gt_num   = extract_number(gt)
    if pred_num is None or gt_num is None:
        return None
    if gt_num == 0:
        correct = abs(pred_num - gt_num) < 1e-9
    else:
        correct = abs(pred_num - gt_num) / abs(gt_num) <= NUMERIC_TOLERANCE
    reason = (
        f"Numeric match: predicted={pred_num}, ground_truth={gt_num}, "
        f"relative_error={abs(pred_num - gt_num) / max(abs(gt_num), 1e-9):.4f}"
    )
    return {"method": "numeric", "correct": correct, "reason": reason}


# ─────────────────────────────────────────────
# 2. String match
# ─────────────────────────────────────────────

def string_match(pred: str, gt: str) -> dict:
    pred_clean = pred.strip().lower()
    gt_clean   = gt.strip().lower()
    correct    = (pred_clean == gt_clean) or (gt_clean in pred_clean)
    reason     = f"String match: predicted='{pred_clean}', ground_truth='{gt_clean}'"
    return {"method": "string_match", "correct": correct, "reason": reason}


# ─────────────────────────────────────────────
# 3. Evaluate a single row
# ─────────────────────────────────────────────

def evaluate_row(row: pd.Series) -> dict:
    pred = str(row.get(COL_PRED, ""))
    gt   = str(row.get(COL_GROUND_TRUTH, ""))
    if not pred or pred.strip() in ("", "nan", "None"):
        return {"method": "numeric", "correct": False, "reason": "Model answer is empty"}
    result = numeric_match(pred, gt)
    if result is not None:
        return result
    return string_match(pred, gt)


# ─────────────────────────────────────────────
# 4. Evaluate a single dataset
# ─────────────────────────────────────────────

def evaluate_dataset(name: str, result_path: str) -> pd.DataFrame:
    logger.log(f"\n{'='*55}")
    logger.log(f"  Dataset: {name}")

    if not Path(result_path).exists():
        logger.log(f"  ⚠️  Result file not found: {result_path}")
        return pd.DataFrame()
    if not Path(GROUND_TRUTH_FILE).exists():
        logger.log(f"  ⚠️  Ground truth file not found: {GROUND_TRUTH_FILE}")
        return pd.DataFrame()

    df = pd.read_csv(result_path)

    for col in [COL_QUESTION_ID, COL_QUESTION, COL_PRED, COL_STATUS]:
        if col not in df.columns:
            logger.log(f"  ❌ Missing column: '{col}' | Available: {list(df.columns)}")
            return pd.DataFrame()

    sheet = GROUND_TRUTH_SHEETS[name]
    try:
        gt_df = pd.read_excel(GROUND_TRUTH_FILE, sheet_name=sheet)[[COL_GT_ID, COL_GT_ANSWER]]
    except Exception as e:
        logger.log(f"  ❌ Failed to read sheet '{sheet}': {e}")
        return pd.DataFrame()

    gt_df = gt_df.rename(columns={COL_GT_ANSWER: COL_GROUND_TRUTH})

    if COL_GROUND_TRUTH in df.columns:
        df = df.drop(columns=[COL_GROUND_TRUTH])
    df = df.merge(gt_df, left_on=COL_QUESTION_ID, right_on=COL_GT_ID, how="left")
    df["dataset"] = name

    total     = len(df)
    n_success = (df[COL_STATUS] == SUCCESS_STATUS).sum()
    n_error   = total - n_success
    logger.log(f"  Total rows   : {total}")
    logger.log(f"  Exec success : {n_success} ({n_success/total*100:.1f}%)")
    logger.log(f"  Exec error   : {n_error} ({n_error/total*100:.1f}%)")

    df_valid = df[df[COL_STATUS] == SUCCESS_STATUS].copy()
    logger.log(f"  Evaluating   : {len(df_valid)} successful rows")

    if df_valid.empty:
        df["eval_method"]  = None
        df["eval_correct"] = None
        df["eval_reason"]  = None
        return df

    eval_results = []
    for i, (_, row) in enumerate(df_valid.iterrows()):
        result = evaluate_row(row)
        eval_results.append(result)
        status = "✅" if result["correct"] else "❌"
        model  = str(row.get(COL_MODEL, ""))[:20]
        mode   = str(row.get(COL_PROMPT_MODE, ""))[:12]
        q      = str(row.get(COL_QUESTION, ""))[:35]
        logger.log(f"  [{i+1:3d}] {status} [{model:<20s}|{mode:<12s}] {q}")

    result_df = pd.DataFrame(eval_results)
    df_valid["eval_method"]  = result_df["method"].values
    df_valid["eval_correct"] = result_df["correct"].values
    df_valid["eval_reason"]  = result_df["reason"].values

    df = df.merge(
        df_valid[[COL_QUESTION_ID, COL_MODEL, COL_PROMPT_MODE,
                  "eval_method", "eval_correct", "eval_reason"]],
        on=[COL_QUESTION_ID, COL_MODEL, COL_PROMPT_MODE],
        how="left"
    )

    return df


# ─────────────────────────────────────────────
# 5. Print + log summary
# ─────────────────────────────────────────────

def df_to_str(df) -> str:
    """Convert a DataFrame to a clean string for logging."""
    buf = io.StringIO()
    df.to_string(buf)
    return buf.getvalue()


def print_summary(all_df: pd.DataFrame):
    sep = "=" * 55
    logger.log(f"\n{sep}")
    logger.log("  📊  EVALUATION SUMMARY REPORT")
    logger.log(sep)

    # ── 1. Execution Success Rate ──
    logger.log("\n" + "─"*55)
    logger.log("  1. EXECUTION SUCCESS RATE")
    logger.log("─"*55)

    total     = len(all_df)
    n_success = (all_df[COL_STATUS] == SUCCESS_STATUS).sum()
    logger.log(f"\n  Overall: {n_success}/{total} = {n_success/total*100:.1f}%")

    if COL_MODEL in all_df.columns:
        logger.log(f"\n  By Model:")
        for model, g in all_df.groupby(COL_MODEL):
            n = (g[COL_STATUS] == SUCCESS_STATUS).sum()
            logger.log(f"    {model}: {n}/{len(g)} = {n/len(g)*100:.1f}%")

    if COL_PROMPT_MODE in all_df.columns:
        logger.log(f"\n  By Prompt Mode:")
        for mode, g in all_df.groupby(COL_PROMPT_MODE):
            n = (g[COL_STATUS] == SUCCESS_STATUS).sum()
            logger.log(f"    {mode}: {n}/{len(g)} = {n/len(g)*100:.1f}%")

    # ── 2. Answer Accuracy ──
    logger.log("\n" + "─"*55)
    logger.log("  2. ANSWER ACCURACY (successful rows only)")
    logger.log("─"*55)

    valid = all_df[all_df["eval_correct"].notna()].copy()
    valid["eval_correct"] = valid["eval_correct"].astype(bool)

    if valid.empty:
        logger.log("\n  ⚠️  No rows to compute accuracy.")
    else:
        overall_acc = valid["eval_correct"].mean() * 100
        logger.log(f"\n  Overall: {overall_acc:.1f}%  ({int(valid['eval_correct'].sum())}/{len(valid)})")

        logger.log(f"\n  By Dataset:")
        by_ds = (
            valid.groupby("dataset")["eval_correct"]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "Accuracy", "sum": "Correct", "count": "Total"})
        )
        by_ds["Accuracy"] = (by_ds["Accuracy"] * 100).round(1).astype(str) + "%"
        logger.log(df_to_str(by_ds))

        if COL_MODEL in valid.columns:
            logger.log(f"\n  By Model:")
            by_model = (
                valid.groupby(COL_MODEL)["eval_correct"]
                .agg(["mean", "sum", "count"])
                .rename(columns={"mean": "Accuracy", "sum": "Correct", "count": "Total"})
                .sort_values("Accuracy", ascending=False)
            )
            by_model["Accuracy"] = (by_model["Accuracy"] * 100).round(1).astype(str) + "%"
            logger.log(df_to_str(by_model))

    # ── 3. Prompt Mode Comparison ──
    logger.log("\n" + "─"*55)
    logger.log("  3. PROMPT MODE COMPARISON")
    logger.log("─"*55)

    if not valid.empty and COL_PROMPT_MODE in valid.columns:
        logger.log(f"\n  Accuracy by Prompt Mode:")
        by_mode = (
            valid.groupby(COL_PROMPT_MODE)["eval_correct"]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "Accuracy", "sum": "Correct", "count": "Total"})
            .sort_values("Accuracy", ascending=False)
        )
        by_mode["Accuracy"] = (by_mode["Accuracy"] * 100).round(1).astype(str) + "%"
        logger.log(df_to_str(by_mode))

    if not valid.empty and COL_MODEL in valid.columns and COL_PROMPT_MODE in valid.columns:
        logger.log(f"\n  Model × Prompt Accuracy Matrix (%):")
        matrix = (
            valid.groupby([COL_MODEL, COL_PROMPT_MODE])["eval_correct"]
            .mean().mul(100).round(1)
            .unstack(COL_PROMPT_MODE)
        )
        logger.log(df_to_str(matrix))

    # ── 4. Python Code Efficiency ──
    logger.log("\n" + "─"*55)
    logger.log("  4. PYTHON CODE EFFICIENCY")
    logger.log("─"*55)

    success_df = all_df[all_df[COL_STATUS] == SUCCESS_STATUS].copy()

    if success_df.empty:
        logger.log("\n  ⚠️  No successful rows to analyze.")
    else:
        # Code lines
        if COL_CODE in success_df.columns:
            success_df["code_lines"] = success_df[COL_CODE].apply(
                lambda x: len([l for l in str(x).split("\n") if l.strip() != ""])
            )
            logger.log(f"\n  Average Lines of Code by Prompt Mode:")
            code_by_mode = (
                success_df.groupby(COL_PROMPT_MODE)["code_lines"]
                .agg(["mean", "min", "max"])
                .rename(columns={"mean": "Avg Lines", "min": "Min", "max": "Max"})
                .round(1)
            )
            logger.log(df_to_str(code_by_mode))

        # Elapsed time
        if COL_ELAPSED in success_df.columns:
            logger.log(f"\n  Response Time by Prompt Mode (seconds):")
            time_by_mode = (
                success_df.groupby(COL_PROMPT_MODE)[COL_ELAPSED]
                .agg(["mean", "min", "max"])
                .rename(columns={"mean": "Avg (s)", "min": "Min (s)", "max": "Max (s)"})
                .round(2)
            )
            logger.log(df_to_str(time_by_mode))

            if COL_MODEL in success_df.columns:
                logger.log(f"\n  Response Time by Model (seconds):")
                time_by_model = (
                    success_df.groupby(COL_MODEL)[COL_ELAPSED]
                    .agg(["mean", "min", "max"])
                    .rename(columns={"mean": "Avg (s)", "min": "Min (s)", "max": "Max (s)"})
                    .round(2)
                    .sort_values("Avg (s)")
                )
                logger.log(df_to_str(time_by_model))

        # Code lines × Prompt mode × Accuracy
        if COL_CODE in success_df.columns and not valid.empty:
            logger.log(f"\n  Avg Lines of Code vs Accuracy by Prompt Mode:")
            merged = valid.copy()
            merged["code_lines"] = merged[COL_CODE].apply(
                lambda x: len([l for l in str(x).split("\n") if l.strip() != ""])
            ) if COL_CODE in merged.columns else None

            if "code_lines" in merged.columns:
                efficiency = merged.groupby(COL_PROMPT_MODE).agg(
                    Accuracy=("eval_correct", lambda x: f"{x.mean()*100:.1f}%"),
                    Avg_Lines=("code_lines", "mean"),
                    Avg_Time=(COL_ELAPSED, "mean") if COL_ELAPSED in merged.columns else ("eval_correct", "count"),
                ).round(2)
                logger.log(df_to_str(efficiency))

    logger.log(f"\n{sep}\n")


# ─────────────────────────────────────────────
# 6. Save JSON summary
# ─────────────────────────────────────────────

def save_summary_json(all_df: pd.DataFrame):
    total     = len(all_df)
    n_success = int((all_df[COL_STATUS] == SUCCESS_STATUS).sum())

    valid = all_df[all_df["eval_correct"].notna()].copy()
    valid["eval_correct"] = valid["eval_correct"].astype(bool)

    summary = {
        "execution": {
            "total":        total,
            "success":      n_success,
            "error":        total - n_success,
            "success_rate": round(n_success / total * 100, 1) if total > 0 else 0,
        },
        "accuracy": {
            "overall": round(valid["eval_correct"].mean() * 100, 1) if not valid.empty else None,
            "correct": int(valid["eval_correct"].sum()) if not valid.empty else 0,
            "total":   len(valid),
        },
        "by_dataset":            {},
        "by_model":              {},
        "by_prompt_mode":        {},
        "model_x_prompt_matrix": {},
        "exec_success_by_model": {},
        "code_efficiency":       {},
    }

    if COL_MODEL in all_df.columns:
        for model, g in all_df.groupby(COL_MODEL):
            n = int((g[COL_STATUS] == SUCCESS_STATUS).sum())
            summary["exec_success_by_model"][model] = {
                "success": n, "total": len(g),
                "rate": round(n / len(g) * 100, 1),
            }

    if not valid.empty:
        for ds, g in valid.groupby("dataset"):
            summary["by_dataset"][ds] = {
                "accuracy": round(g["eval_correct"].mean() * 100, 1),
                "correct":  int(g["eval_correct"].sum()), "total": len(g),
            }
        if COL_MODEL in valid.columns:
            for model, g in valid.groupby(COL_MODEL):
                summary["by_model"][model] = {
                    "accuracy": round(g["eval_correct"].mean() * 100, 1),
                    "correct":  int(g["eval_correct"].sum()), "total": len(g),
                }
        if COL_PROMPT_MODE in valid.columns:
            for mode, g in valid.groupby(COL_PROMPT_MODE):
                summary["by_prompt_mode"][mode] = {
                    "accuracy": round(g["eval_correct"].mean() * 100, 1),
                    "correct":  int(g["eval_correct"].sum()), "total": len(g),
                }
        if COL_MODEL in valid.columns and COL_PROMPT_MODE in valid.columns:
            for model, mg in valid.groupby(COL_MODEL):
                summary["model_x_prompt_matrix"][model] = {}
                for mode, g in mg.groupby(COL_PROMPT_MODE):
                    summary["model_x_prompt_matrix"][model][mode] = round(
                        g["eval_correct"].mean() * 100, 1
                    )

    # Code efficiency
    success_df = all_df[all_df[COL_STATUS] == SUCCESS_STATUS]
    if not success_df.empty:
        if COL_CODE in success_df.columns and COL_PROMPT_MODE in success_df.columns:
            code_lines = success_df.copy()
            code_lines["code_lines"] = code_lines[COL_CODE].apply(
                lambda x: len([l for l in str(x).split("\n") if l.strip() != ""])
            )
            for mode, g in code_lines.groupby(COL_PROMPT_MODE):
                summary["code_efficiency"][mode] = {
                    "avg_lines": round(g["code_lines"].mean(), 1),
                    "avg_elapsed_sec": round(g[COL_ELAPSED].mean(), 2) if COL_ELAPSED in g.columns else None,
                }

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.log(f"💾 JSON summary saved: {OUTPUT_JSON}")


# ─────────────────────────────────────────────
# 7. Entry point
# ─────────────────────────────────────────────

def main():
    logger.log("🚀 Starting evaluation...\n")

    all_dfs = []
    for name in RESULT_FILES:
        df = evaluate_dataset(name, result_path=RESULT_FILES[name])
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        logger.log("\n⚠️  No datasets evaluated. Check file paths and column name config above.")
        return

    all_df = pd.concat(all_dfs, ignore_index=True)

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    logger.log(f"\n💾 Detailed results saved: {OUTPUT_CSV}")

    print_summary(all_df)
    save_summary_json(all_df)

    # Save txt report
    logger.save(OUTPUT_TXT)

    logger.log("✅ Evaluation complete!")


if __name__ == "__main__":
    main()