from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    dataset_id: str = Field(..., description="Selected dataset identifier")
    question: str = Field(..., min_length=2)
    model_id: str = Field(..., description="OpenRouter model id")
    prompt_mode: Literal[
        "baseline",
        "structured",
        "CoT",
    ]
    temperature: float = Field(0.2, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    dataset_id: str
    question: str
    model_id: str
    prompt_mode: str
    final_answer: str
    python_code: str
    execution_stdout: str
    execution_error: Optional[str] = None
    raw_model_output: str
    trace_count: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)
