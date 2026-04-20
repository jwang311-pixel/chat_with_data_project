from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import PROMPTS_DIR


class PromptService:
    def __init__(self) -> None:
        self.base_system = (PROMPTS_DIR / "system.txt").read_text(encoding="utf-8")

        self.mode_templates = {
            "baseline": (PROMPTS_DIR / "baseline.txt").read_text(encoding="utf-8"),
            "structured": (PROMPTS_DIR / "structured.txt").read_text(encoding="utf-8"),
            "CoT": (PROMPTS_DIR / "CoT.txt").read_text(encoding="utf-8"),
        }

        self.refine_template = (PROMPTS_DIR / "refine_review.txt").read_text(encoding="utf-8")

        # Optional files. If they exist, we use them.
        self.code_template = self._read_optional_prompt("code_generation.txt")
        self.answer_template = self._read_optional_prompt("answer_generation.txt")

    def _read_optional_prompt(self, filename: str) -> str:
        path = PROMPTS_DIR / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def _safe_json(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(value)

    def dataset_context(self, dataset: dict) -> str:
        preview = dataset.get("preview", [])
        return (
            f"Dataset name: {dataset.get('display_name', '')}\n"
            f"Dataset id: {dataset.get('dataset_id', '')}\n"
            f"Filename: {dataset.get('filename', '')}\n"
            f"Rows: {dataset.get('row_count', '')}\n"
            f"Columns: {dataset.get('column_count', '')}\n"
            f"Column names: {', '.join(dataset.get('columns', []))}\n"
            f"Numeric columns: {', '.join(dataset.get('numeric_columns', []))}\n"
            f"Text columns: {', '.join(dataset.get('text_columns', []))}\n"
            f"Missing values: {self._safe_json(dataset.get('missing_values', {}))}\n"
            f"Preview rows:\n{self._safe_json(preview)}\n"
        )

    def build_messages(
        self,
        *,
        dataset: dict,
        question: str,
        mode: str,
        candidate_index: int | None = None,
        previous: dict | None = None,
    ) -> list[dict]:
        """
        Backward-compatible builder.

        - If previous is provided, this is the old self-refine style prompt.
        - Otherwise, this is the old one-shot prompt style for the selected mode.
        """
        if previous is not None:
            body = self.refine_template.format(
                dataset_context=self.dataset_context(dataset),
                question=question,
                previous_code=previous.get("python_code", ""),
                previous_answer=previous.get("final_answer", ""),
                previous_stdout=previous.get("execution_stdout", ""),
                previous_error=previous.get("execution_error", ""),
            )
        else:
            template = self.mode_templates.get(mode, self.mode_templates["baseline"])
            body = template.format(
                dataset_context=self.dataset_context(dataset),
                question=question,
                candidate_index=candidate_index or 1,
            )

        return [
            {"role": "system", "content": self.base_system},
            {"role": "user", "content": body},
        ]

    def build_code_messages(
        self,
        *,
        dataset: dict,
        question: str,
        mode: str,
        candidate_index: int = 1,
        previous: dict | None = None,
        stage: str = "code",
    ) -> list[dict]:
        """
        Stage 1: ask the model to generate Python code only.
        """
        dataset_context = self.dataset_context(dataset)

        if self.code_template:
            body = self.code_template.format(
                dataset_context=dataset_context,
                question=question,
                mode=mode,
                candidate_index=candidate_index,
                stage=stage,
            )
        else:
            template = self.mode_templates.get(mode, self.mode_templates["baseline"])
            body = template.format(
                dataset_context=dataset_context,
                question=question,
                candidate_index=candidate_index,
            )
            body += (
                "\n\nIMPORTANT:\n"
                "- Return JSON only.\n"
                '- The JSON must contain a "python_code" field.\n'
                '- The python_code must be valid pandas code.\n'
                '- The code must assign the final answer to a variable named result.\n'
                "- Do not include explanations outside JSON.\n"
            )

        return [
            {"role": "system", "content": self.base_system},
            {"role": "user", "content": body},
        ]

    def build_answer_messages(
        self,
        *,
        dataset: dict,
        question: str,
        mode: str,
        previous: dict,
        candidate_index: int = 1,
        stage: str = "answer",
    ) -> list[dict]:
        """
        Stage 2: ask the model to turn the true execution result into a final answer.
        """
        dataset_context = self.dataset_context(dataset)

        python_code = previous.get("python_code", "")
        execution_result = previous.get("execution_result", "")
        execution_stdout = previous.get("execution_stdout", "")
        execution_error = previous.get("execution_error", "")

        if self.answer_template:
            body = self.answer_template.format(
                dataset_context=dataset_context,
                question=question,
                mode=mode,
                candidate_index=candidate_index,
                python_code=python_code,
                execution_result=self._safe_json(execution_result),
                execution_stdout=execution_stdout,
                execution_error=execution_error,
                stage=stage,
            )
        else:
            body = (
                f"Dataset context:\n{dataset_context}\n"
                f"Question:\n{question}\n\n"
                f"Python code:\n{python_code}\n\n"
                f"Execution result:\n{self._safe_json(execution_result)}\n\n"
                f"stdout:\n{execution_stdout}\n\n"
                f"error:\n{execution_error}\n\n"
                "Rules:\n"
                "- Answer the question using ONLY the execution result.\n"
                "- Do not invent any numbers.\n"
                "- If execution failed, say so clearly.\n"
                "- Return JSON only with key:\n"
                '  { "final_answer": "..." }\n'
            )

        return [
            {"role": "system", "content": self.base_system},
            {"role": "user", "content": body},
        ]