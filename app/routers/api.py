from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from app.core.config import MODELS_FILE, PROMPT_MODES_FILE, load_json
from app.schemas import ChatRequest
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api")

from app.services.chat_service import ChatService
from app.services.data_service import DataService
from app.services.openrouter_client import OpenRouterClient
from app.services.sandbox_service import SandboxService
from app.services.prompt_service import PromptService

data_service = DataService()
llm = OpenRouterClient()
sandbox = SandboxService()
prompt_service = PromptService()

chat_service = ChatService(
    data_service=data_service,
    llm=llm,
    sandbox=sandbox,
    prompt_service=prompt_service,
)

data_service = chat_service.data_service


@router.get("/datasets")
def list_datasets():
    return data_service.list_datasets()


@router.get("/models")
def list_models():
    return load_json(MODELS_FILE)


@router.get("/prompt-modes")
def list_prompt_modes():
    return load_json(PROMPT_MODES_FILE)


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    return await data_service.save_upload(file)


from fastapi import HTTPException
from fastapi.responses import JSONResponse
import traceback

@router.post("/chat")
def chat(req: ChatRequest):
    try:
        return chat_service.answer(req)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(e),
                "traceback": tb,
            },
        )
