"""
llm_judge.py
=============
用 LLM-as-judge 对结果进行三个维度的评分：
  - Clarity:     答案是否清晰易懂 (1-5)
  - Relevance:   答案是否切题 (1-5)
  - Correctness: 和 ground truth 对比是否正确 (1-5)

输出：results/llm_judge_report.csv

运行方式：python llm_judge.py
需要：pip install pandas openpyxl requests
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import requests

# ── 配置 ──────────────────────────────────────────────────────────────────────

RESULTS_DIR       = Path("results")
GROUND_TRUTH_FILE = Path("evaluation/questions/ground_truth.xlsx")
OUTPUT_FILE       = RESULTS_DIR / "llm_judge_report.csv"

JUDGE_MODEL = "openai/gpt-3.5-turbo"

SHEET_MAP = {
    "customer_service":    "customer_service",
    "finance_expenses":    "finance_analysis",
    "marketing_campaigns": "marketing_campaign",
    "hr_employees":        "HR_employees",
    "sales_orders":        "sales_order",
}

OUTPUT_FIELDS = [
    "dataset_id",
    "question_id",
    "question",
    "model_id",
    "prompt_mode",
    "status",
    "final_answer",
    "ground_truth_answer",
    "clarity",
    "clarity_reason",
    "relevance",
    "relevance_reason",
    "correctness",
    "correctness_reason",
    "avg_score",
]

# ── 读取 API key ───────────────────────────────────────────────────────────────

def load_api_key() -> str:
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    import os
    return os.getenv("OPENROUTER_API_KEY", "")

API_KEY = load_api_key()

# ── 读取 ground truth ─────────────────────────────────────────────────────────

def load_ground_truth() -> Dict[str, Dict[str, str]]:
    """返回 {dataset_id: {question_id: answer}}"""
    wb_sheets = pd.read_excel(GROUND_TRUTH_FILE, sheet_name=None)
    gt = {}
    for dataset_id, sheet_name in SHEET_MAP.items():
        if sheet_name not in wb_sheets:
            print(f"[WARN] sheet '{sheet_name}' not found")
            continue
        df = wb_sheets[sheet_name]
        df.columns = [c.strip().lower() for c in df.columns]
        if "question_id" not in df.columns or "answer" not in df.columns:
            print(f"[WARN] sheet '{sheet_name}' missing question_id or answer column")
            continue
        gt[dataset_id] = dict(zip(df["question_id"].astype(str), df["answer"].astype(str)))
    return gt

# ── LLM 打分 ─────────────────────────────────────────────────────────────────

def call_judge(question: str, final_answer: str, ground_truth: str) -> Dict[str, Any]:
    """
    调用 LLM 对答案进行三个维度打分，返回分数和理由。
    """
    prompt = f"""You are an objective evaluator for a data analysis system.

Evaluate the system's answer on THREE dimensions, each scored 1-5:

1. Clarity (1-5): Is the answer clear and easy to understand?
   1=very confusing, 3=somewhat clear, 5=very clear and well-explained

2. Relevance (1-5): Does the answer directly address the question asked?
   1=completely off-topic, 3=partially relevant, 5=directly answers the question

3. Correctness (1-5): How accurate is the answer compared to the ground truth?
   1=completely wrong, 2=major errors, 3=partially correct, 4=minor errors, 5=correct

Question: {question}

Ground truth answer: {ground_truth}

System's answer: {final_answer}

Return JSON only, no other text:
{{
  "clarity": <1-5>,
  "clarity_reason": "<one sentence>",
  "relevance": <1-5>,
  "relevance_reason": "<one sentence>",
  "correctness": <1-5>,
  "correctness_reason": "<one sentence>"
}}"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 300,
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  [429] waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"  [API error] {resp.status_code}: {resp.text[:100]}")
                return _empty_scores(f"API error {resp.status_code}")

            content = resp.json()["choices"][0]["message"]["content"].strip()
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(content[start:end+1])
                return {
                    "clarity":             int(parsed.get("clarity", 0)),
                    "clarity_reason":      str(parsed.get("clarity_reason", "")),
                    "relevance":           int(parsed.get("relevance", 0)),
                    "relevance_reason":    str(parsed.get("relevance_reason", "")),
                    "correctness":         int(parsed.get("correctness", 0)),
                    "correctness_reason":  str(parsed.get("correctness_reason", "")),
                }
        except Exception as e:
            print(f"  [error] {e}")
            time.sleep(5)

    return _empty_scores("Failed after 3 retries")


def _empty_scores(reason: str) -> Dict[str, Any]:
    return {
        "clarity": 0, "clarity_reason": reason,
        "relevance": 0, "relevance_reason": reason,
        "correctness": 0, "correctness_reason": reason,
    }

# ── 写入报告 ──────────────────────────────────────────────────────────────────

def append_report(path: Path, record: Dict[str, Any], first: bool) -> None:
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        if first:
            writer.writeheader()
        safe = {k: ("" if record.get(k) is None else str(record.get(k, ""))) for k in OUTPUT_FIELDS}
        writer.writerow(safe)

# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading ground truth...")
    gt = load_ground_truth()
    for ds, qa in gt.items():
        print(f"  {ds}: {len(qa)} questions")

    result_files = sorted(RESULTS_DIR.glob("*_results.csv"))
    if not result_files:
        print(f"No result files found in {RESULTS_DIR}/")
        return

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    total = 0
    first_write = True

    for result_file in result_files:
        dataset_id = result_file.stem.replace("_results", "")
        df = pd.read_csv(result_file, encoding="utf-8-sig")
        dataset_gt = gt.get(dataset_id, {})

        print(f"\n=== {result_file.name} ({len(df)} rows)")

        for _, row in df.iterrows():
            total += 1
            status       = str(row.get("status", ""))
            question     = str(row.get("question", "")).strip()
            question_id  = str(row.get("question_id", "")).strip()
            final_answer = str(row.get("final_answer", "")).strip()
            model_id     = str(row.get("model_id", ""))
            prompt_mode  = str(row.get("prompt_mode", ""))

            ground_truth_answer = dataset_gt.get(question_id, "")

            print(f"  [{total}] {dataset_id} | {question_id} | {model_id[:25]} | {prompt_mode}")

            # 执行失败的行直接给 0 分
            if status != "ok" or not final_answer or final_answer in ("nan", "None", ""):
                scores = _empty_scores("Execution failed or no answer")
            else:
                scores = call_judge(
                    question=question,
                    final_answer=final_answer,
                    ground_truth=ground_truth_answer,
                )
                time.sleep(1)  # 避免限速

            avg = round(
                (scores["clarity"] + scores["relevance"] + scores["correctness"]) / 3, 2
            ) if scores["clarity"] > 0 else 0

            print(f"    clarity={scores['clarity']} relevance={scores['relevance']} correctness={scores['correctness']} avg={avg}")

            record = {
                "dataset_id":          dataset_id,
                "question_id":         question_id,
                "question":            question,
                "model_id":            model_id,
                "prompt_mode":         prompt_mode,
                "status":              status,
                "final_answer":        final_answer,
                "ground_truth_answer": ground_truth_answer,
                "avg_score":           avg,
                **scores,
            }

            append_report(OUTPUT_FILE, record, first=first_write)
            first_write = False

    print(f"\nDone. {total} rows evaluated.")
    print(f"Report saved to {OUTPUT_FILE}")

    # ── 打印汇总 ──────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  LLM JUDGE SUMMARY")
    print("="*55)

    report_df = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
    scored = report_df[report_df["avg_score"] > 0]

    if scored.empty:
        print("No scored rows.")
        return

    print(f"\nOverall (n={len(scored)}):")
    print(f"  Clarity:     {scored['clarity'].mean():.2f}/5")
    print(f"  Relevance:   {scored['relevance'].mean():.2f}/5")
    print(f"  Correctness: {scored['correctness'].mean():.2f}/5")
    print(f"  Average:     {scored['avg_score'].mean():.2f}/5")

    print(f"\nBy Model:")
    for model, g in scored.groupby("model_id"):
        print(f"  {model[:35]}: clarity={g['clarity'].mean():.2f} relevance={g['relevance'].mean():.2f} correctness={g['correctness'].mean():.2f} avg={g['avg_score'].mean():.2f}")

    print(f"\nBy Prompt Mode:")
    for mode, g in scored.groupby("prompt_mode"):
        print(f"  {mode}: clarity={g['clarity'].mean():.2f} relevance={g['relevance'].mean():.2f} correctness={g['correctness'].mean():.2f} avg={g['avg_score'].mean():.2f}")


if __name__ == "__main__":
    main()