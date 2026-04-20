from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


BASE_URL = "http://127.0.0.1:8000"

QUESTION_DIR = Path("evaluation/questions")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "openai/gpt-3.5-turbo",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-coder:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

PROMPT_MODES = [
    "baseline",
    "structured",
    "CoT",
]

DATASET_MAP = {
    "finance_expenses_questions.csv": "finance_expenses",
    "customer_service_questions.csv": "customer_service",
    "sales_orders_questions.csv": "sales_orders",
    "hr_employees_questions.csv": "hr_employees",
    "marketing_campaigns_questions.csv": "marketing_campaigns",
}

CSV_FIELDS = [
    "dataset_file",
    "dataset_id",
    "question_id",
    "question",
    "model_id",
    "prompt_mode",
    "status",
    "final_answer",
    "python_code",
    "execution_result",
    "execution_stdout",
    "execution_error",
    "trace_count",
    "elapsed_sec",
    "error",
    "started_at",
]


def read_questions(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_dataset_id(question_file: Path) -> str:
    if question_file.name in DATASET_MAP:
        return DATASET_MAP[question_file.name]
    stem = question_file.stem
    if stem.endswith("_questions"):
        return stem[: -len("_questions")]
    return stem


def get_question_id(row: Dict[str, str], idx: int) -> str:
    for key in ("id", "question_id", "qid", "qid_num"):
        if key in row and row[key]:
            return str(row[key]).strip()
    return f"q{idx:04d}"


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def call_chat_api(
    dataset_id: str,
    question: str,
    model_id: str,
    prompt_mode: str,
    temperature: float = 0.2,
    timeout: int = 300,
) -> Dict[str, Any]:
    payload = {
        "dataset_id": dataset_id,
        "question": question,
        "model_id": model_id,
        "prompt_mode": prompt_mode,
        "temperature": temperature,
    }
    resp = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def append_csv(path: Path, record: Dict[str, Any]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        safe_record = {k: safe_text(record.get(k)) for k in CSV_FIELDS}
        writer.writerow(safe_record)


def main() -> None:
    question_files = sorted(QUESTION_DIR.glob("*_questions.csv"))
    if not question_files:
        raise FileNotFoundError(f"No question files found in {QUESTION_DIR}")

    total_runs = 0

    for question_file in question_files:
        dataset_id = get_dataset_id(question_file)
        questions = read_questions(question_file)

        output_file = RESULTS_DIR / f"{dataset_id}_results.csv"

        print(f"\n=== Dataset file: {question_file.name}")
        print(f"Dataset ID: {dataset_id}")
        print(f"Questions: {len(questions)}")
        print(f"Output file: {output_file}")

        for q_idx, row in enumerate(questions, start=1):
            question_id = get_question_id(row, q_idx)
            question_text = row.get("question", "").strip()

            if not question_text:
                print(f"[SKIP] {question_id}: empty question")
                continue

            for model_id in MODELS:
                for prompt_mode in PROMPT_MODES:
                    total_runs += 1
                    print(
                        f"[RUN {total_runs}] "
                        f"{dataset_id} | {question_id} | {model_id} | {prompt_mode}"
                    )

                    started = time.time()
                    record: Dict[str, Any] = {
                        "dataset_file": question_file.name,
                        "dataset_id": dataset_id,
                        "question_id": question_id,
                        "question": question_text,
                        "model_id": model_id,
                        "prompt_mode": prompt_mode,
                        "started_at": started,
                    }

                    try:
                        data = call_chat_api(
                            dataset_id=dataset_id,
                            question=question_text,
                            model_id=model_id,
                            prompt_mode=prompt_mode,
                        )

                        record.update(
                            {
                                "status": "ok",
                                "final_answer": data.get("final_answer"),
                                "python_code": data.get("python_code"),
                                "execution_result": data.get("execution_result"),
                                "execution_stdout": data.get("execution_stdout"),
                                "execution_error": data.get("execution_error"),
                                "trace_count": data.get("trace_count"),
                            }
                        )

                        print("  answer:", safe_text(record["final_answer"])[:160])

                    except Exception as e:
                        record.update(
                            {
                                "status": "error",
                                "error": str(e),
                            }
                        )
                        print("  ERROR:", e)

                    record["elapsed_sec"] = round(time.time() - started, 3)
                    append_csv(output_file, record)

    print(f"\nDone. {total_runs} runs total. Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()