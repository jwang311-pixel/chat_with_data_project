from __future__ import annotations

import json
import re
import traceback
from collections import Counter
from typing import Any, List, Optional

from app.schemas import ChatRequest
print("LOADED:", __file__, flush=True)

def _make_json_safe(obj: Any) -> Any:
    """
    Recursively convert common non-JSON-safe values into JSON-safe ones.
    """
    try:
        import math
        import numpy as np
        import pandas as pd
    except Exception:
        math = None
        np = None
        pd = None

    if obj is None:
        return None

    if isinstance(obj, (str, int, bool)):
        return obj

    if isinstance(obj, float):
        if math is not None and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    if np is not None:
        if isinstance(obj, np.floating):
            value = float(obj)
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        if isinstance(obj, np.integer):
            return int(obj)

    if pd is not None:
        if isinstance(obj, pd.DataFrame):
            cleaned = obj.replace([float("inf"), float("-inf")], None)
            cleaned = cleaned.where(pd.notna(cleaned), None)
            return cleaned.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            cleaned = obj.replace([float("inf"), float("-inf")], None)
            cleaned = cleaned.where(pd.notna(cleaned), None)
            return cleaned.to_list()

    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in obj]

    return str(obj)


def _strip_code_fences(text: str) -> str:
    """
    Remove markdown code fences if the model returned them.
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _extract_first_json_object(text: str) -> Optional[dict]:
    """
    Try to extract the first JSON object from a model response.
    Returns None if parsing fails.
    """
    if not text:
        return None

    text = text.strip()

    # 1) direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2) look for any {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def _normalize_answer(text: str) -> str:
    """
    Normalize an answer string for majority voting.
    """
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\-\:,/%$]", "", text)
    return text


def _safe_text(value: Any) -> str:
    """
    Convert any value to a readable string for prompting.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(_make_json_safe(value), ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


class ChatService:
    """
    Two-stage chat service:

    Stage 1:
      user question + dataset schema -> LLM generates python_code

    Stage 2:
      python_code + sandbox execution result -> LLM generates final_answer
    """

    def __init__(self, data_service, llm, sandbox, prompt_service):
        self.data_service = data_service
        self.llm = llm
        self.sandbox = sandbox
        self.prompt_service = prompt_service

    def answer(self, req: ChatRequest) -> dict:
        print("ANSWER METHOD HIT", flush=True)
        try:
            dataset = self.data_service.get_dataset_by_id(req.dataset_id)

            mode = getattr(req, "prompt_mode", "baseline") or "baseline"
            if hasattr(self.prompt_service, "mode_templates") and mode not in self.prompt_service.mode_templates:
                mode = "baseline"

            run = self._run_two_stage_candidate(req=req, dataset=dataset, candidate_index=1)

            response = {
                "dataset_id": req.dataset_id,
                "question": req.question,
                "model_id": req.model_id,
                "prompt_mode": req.prompt_mode,
                "python_code": run["python_code"],
                "execution_stdout": run["execution_stdout"],
                "execution_error": run["execution_error"],
                "execution_result": run["execution_result"],
                "final_answer": run["final_answer"],
                "trace_count": 2,
                "candidate_traces": [run],
            }
            return _make_json_safe(response)

        except Exception as e:
            print("🔥 CHAT ERROR:", repr(e), flush=True)
            traceback.print_exc()
            raise

    def _run_two_stage_candidate(self, req, dataset: dict, candidate_index: int = 1) -> dict:
        print("RUN TWO STAGE HIT", flush=True)
        """
        Stage 1: generate python code.
        Stage 2: execute code.
        Stage 3: ask LLM to summarize the true execution result.
        """
        code_messages = self._build_code_messages(
            req=req,
            dataset=dataset,
            candidate_index=candidate_index,
        )

        raw_code = self.llm.chat(
            model=req.model_id,
            messages=code_messages,
            temperature=getattr(req, "temperature", 0.2) or 0.2,
        )

        code_json = self._extract_json(raw_code)
        if code_json and code_json.get("python_code"):
            python_code = str(code_json.get("python_code")).strip()
            code_reason = str(code_json.get("short_reason") or code_json.get("explanation") or "").strip()
        else:
            python_code = _strip_code_fences(str(raw_code)).strip()
            code_reason = ""

        if not python_code:
            return {
                "candidate_index": candidate_index,
                "python_code": "",
                "code_reason": code_reason,
                "execution_stdout": "",
                "execution_error": "LLM did not return python_code.",
                "execution_result": None,
                "final_answer": "I could not generate Python code for this question.",
                "raw_code_output": str(raw_code),
                "raw_answer_output": "",
            }

        execution = self.sandbox.execute(
            csv_path=dataset["path"],
            python_code=python_code,
        )

        # Debug: see what sandbox returned
        print("EXECUTION OBJECT:", execution)
        print("EXECUTION TYPE:", type(execution))
        if isinstance(execution, dict):
            print("EXECUTION KEYS:", list(execution.keys()))
        else:
            print("EXECUTION DIR:", dir(execution))

        execution_stdout = self._read_field(execution, "stdout", default="")
        execution_error = self._read_field(execution, "error", default="")

        # Try multiple common field names
        execution_result = self._read_field(execution, "result", default=None)
        if execution_result is None:
            execution_result = self._read_field(execution, "result_repr", default=None)
        if execution_result is None:
            execution_result = self._read_field(execution, "output", default=None)
        if execution_result is None:
            execution_result = self._read_field(execution, "value", default=None)

        print("EXECUTION RESULT:", execution_result)

        answer_messages = self._build_answer_messages(
            req=req,
            dataset=dataset,
            python_code=python_code,
            execution_stdout=execution_stdout,
            execution_error=execution_error,
            execution_result=execution_result,
            candidate_index=candidate_index,
        )

        raw_answer = self.llm.chat(
            model=req.model_id,
            messages=answer_messages,
            temperature=getattr(req, "temperature", 0.2) or 0.2,
        )

        print("RAW ANSWER:", raw_answer)

        answer_json = self._extract_json(raw_answer)
        if answer_json and isinstance(answer_json, dict):
            final_answer = str(
                answer_json.get("final_answer")
                or answer_json.get("answer")
                or answer_json.get("response")
                or ""
            ).strip()
        else:
            final_answer = _strip_code_fences(str(raw_answer)).strip()

        if not final_answer:
            final_answer = "I could not produce a final answer from the execution result."

        print("FINAL ANSWER:", final_answer)

        return {
            "candidate_index": candidate_index,
            "python_code": python_code,
            "code_reason": code_reason,
            "execution_stdout": execution_stdout,
            "execution_error": execution_error,
            "execution_result": _make_json_safe(execution_result),
            "final_answer": final_answer,
            "raw_code_output": str(raw_code),
            "raw_answer_output": str(raw_answer),
        }

    def _refine_answer(self, req, dataset: dict, first_run: dict) -> str:
        """
        Optional refinement step for self_refine:
        take the first answer and ask the model to review/correct it using the same execution result.
        """
        refinement_messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful data analyst. Review the previous answer "
                    "and correct any mistakes using ONLY the execution result."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{req.question}\n\n"
                    f"Dataset:\n{_safe_text(dataset)}\n\n"
                    f"Python code:\n{first_run.get('python_code', '')}\n\n"
                    f"Execution result:\n{_safe_text(first_run.get('execution_result'))}\n\n"
                    f"stdout:\n{_safe_text(first_run.get('execution_stdout'))}\n\n"
                    f"error:\n{_safe_text(first_run.get('execution_error'))}\n\n"
                    f"Previous answer:\n{first_run.get('final_answer', '')}\n\n"
                    "Return JSON only:\n"
                    "{{\n"
                    '  "final_answer": "..."\n'
                    "}}"
                ),
            },
        ]

        raw = self.llm.chat(
            model=req.model_id,
            messages=refinement_messages,
            temperature=getattr(req, "temperature", 0.2) or 0.2,
        )

        parsed = _extract_first_json_object(_strip_code_fences(str(raw)))
        if parsed and parsed.get("final_answer"):
            return str(parsed["final_answer"]).strip()

        fallback = _strip_code_fences(str(raw)).strip()
        return fallback or first_run.get("final_answer", "")

    def _majority_vote_answer(self, answers: List[str]) -> str:
        """
        Majority vote over normalized answers, then return the first raw answer
        matching the winning normalized form.
        """
        if not answers:
            return ""

        normalized = [_normalize_answer(a) for a in answers if str(a).strip()]
        if not normalized:
            return str(answers[0] or "").strip()

        counts = Counter(normalized)
        winner_norm, _ = counts.most_common(1)[0]

        for raw in answers:
            if _normalize_answer(raw) == winner_norm:
                return str(raw).strip()

        return str(answers[0]).strip()

    def _build_code_messages(self, req, dataset: dict, candidate_index: int = 1):
        """
        Build the Stage-1 prompt for code generation.
        """
        previous = {"candidate_index": candidate_index}

        builder = getattr(self.prompt_service, "build_code_messages", None)
        if callable(builder):
            try:
                return builder(
                    dataset=dataset,
                    question=req.question,
                    mode=req.prompt_mode,
                    candidate_index=candidate_index,
                    previous=previous,
                    stage="code",
                )
            except TypeError:
                pass

        builder = getattr(self.prompt_service, "build_messages", None)
        if callable(builder):
            try:
                return builder(
                    dataset=dataset,
                    question=req.question,
                    mode=req.prompt_mode,
                    candidate_index=candidate_index,
                    previous=previous,
                )
            except TypeError:
                pass

        dataset_schema = self._dataset_schema_text(dataset)
        preview_text = _safe_text(dataset.get("preview"))
        system_prompt = (
            "You are a data analysis assistant.\n"
            "Generate ONLY valid Python/pandas code.\n"
            "The code must assign the final answer to a variable named result.\n"
            "Do not explain. Do not use markdown. Do not use code fences.\n"
            "Return JSON only with keys:\n"
            '  - "python_code"\n'
            '  - "short_reason"\n'
        )

        if candidate_index > 1:
            system_prompt += (
                f"\nThis is candidate trace #{candidate_index}. "
                "Make the reasoning slightly independent from other traces."
            )

        user_prompt = (
            f"Question:\n{req.question}\n\n"
            f"Prompt mode:\n{req.prompt_mode}\n\n"
            f"Dataset schema:\n{dataset_schema}\n\n"
            f"Dataset preview:\n{preview_text}\n\n"
            "Return JSON only."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_answer_messages(
        self,
        req,
        dataset: dict,
        python_code: str,
        execution_stdout: Any,
        execution_error: Any,
        execution_result: Any,
        candidate_index: int = 1,
    ):
        """
        Build the Stage-2 prompt for natural language answer generation.
        """
        previous = {
            "python_code": python_code,
            "execution_stdout": execution_stdout,
            "execution_error": execution_error,
            "execution_result": execution_result,
            "candidate_index": candidate_index,
        }

        builder = getattr(self.prompt_service, "build_answer_messages", None)
        if callable(builder):
            try:
                return builder(
                    dataset=dataset,
                    question=req.question,
                    mode=req.prompt_mode,
                    previous=previous,
                    candidate_index=candidate_index,
                    stage="answer",
                )
            except TypeError:
                pass

        builder = getattr(self.prompt_service, "build_messages", None)
        if callable(builder):
            try:
                return builder(
                    dataset=dataset,
                    question=req.question,
                    mode=req.prompt_mode,
                    previous=previous,
                )
            except TypeError:
                pass

        system_prompt = (
            "You are a data analyst assistant.\n"
            "Answer the question using ONLY the executed result.\n"
            "Do not invent numbers. If the result is empty or there is an error, say so clearly.\n"
            "Return JSON only with key:\n"
            '  - "final_answer"\n'
        )

        user_prompt = (
            f"Question:\n{req.question}\n\n"
            f"Prompt mode:\n{req.prompt_mode}\n\n"
            f"Python code:\n{python_code}\n\n"
            f"Execution result:\n{_safe_text(execution_result)}\n\n"
            f"stdout:\n{_safe_text(execution_stdout)}\n\n"
            f"error:\n{_safe_text(execution_error)}\n\n"
            "Return JSON only."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _dataset_schema_text(self, dataset: dict) -> str:
        columns = dataset.get("columns") or []
        numeric_columns = dataset.get("numeric_columns") or []
        text_columns = dataset.get("text_columns") or []
        missing_values = dataset.get("missing_values") or {}

        parts = [
            f"dataset_id: {dataset.get('dataset_id', '')}",
            f"filename: {dataset.get('filename', '')}",
            f"display_name: {dataset.get('display_name', '')}",
            f"row_count: {dataset.get('row_count', '')}",
            f"column_count: {dataset.get('column_count', '')}",
            f"columns: {', '.join(columns) if columns else '(unknown)'}",
            f"numeric_columns: {', '.join(numeric_columns) if numeric_columns else '(none)'}",
            f"text_columns: {', '.join(text_columns) if text_columns else '(none)'}",
            f"missing_values: {json.dumps(_make_json_safe(missing_values), ensure_ascii=False)}",
        ]
        return "\n".join(parts)

    def _extract_json(self, raw: Any) -> dict:
        """
        Extract a JSON object from raw model output.
        """
        if isinstance(raw, dict):
            return raw
        parsed = _extract_first_json_object(_strip_code_fences(str(raw)))
        return parsed or {}

    def _read_field(self, obj: Any, key: str, default: Any = None) -> Any:
        """
        Read a field from either a dict-like execution object or an object with attributes.
        """
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)