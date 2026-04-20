# Chat With Your Data

A minimal FastAPI project for uploading a CSV, asking natural-language questions, generating Pandas code with an OpenRouter model, and executing that code in a separate sandbox runner.

## Included
- FastAPI backend
- Plain HTML/CSS/JS frontend
- CSV upload support
- OpenRouter API integration
- Five prompt modes
- Four preconfigured model options
- Separate sandbox runner for generated Python code
- Sample datasets and matching question files

## Quick start
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in `OPENROUTER_API_KEY`.
3. Start the app:

```bash
uvicorn app.main:app --reload
```

4. Open `http://127.0.0.1:8000`.

## Project layout
- `app/main.py` - FastAPI entry point
- `app/routers/api.py` - HTTP endpoints
- `app/services/chat_service.py` - full question-answering flow
- `app/services/prompt_service.py` - prompt assembly
- `app/services/openrouter_client.py` - OpenRouter API wrapper
- `app/services/sandbox_service.py` - calls the runner process
- `app/utils/csv_utils.py` - resilient CSV loading
- `sandbox/runner.py` - isolated code execution
- `frontend/` - simple web UI
- `config/` - model and prompt mode lists
- `evaluation/questions/` - provided question files

## Pipeline
1. The user selects or uploads a CSV.
2. The backend reads the schema and preview rows.
3. The selected prompt mode shapes the LLM request.
4. The LLM returns JSON with `python_code` and `final_answer`.
5. The sandbox runner executes the generated Pandas code against `df`.
6. The app returns the answer, code, stdout, and any runtime error.

## Sandbox note
This is a minimal sandbox for a class project or prototype. It blocks imports through restricted builtins and runs code in a separate process with basic resource limits.
For production, use stronger isolation such as Docker, gVisor, or another hardened execution environment.

## Best files to edit later
- Prompt behavior: `app/prompts/`
- Model list: `config/models.json`
- Execution limits: `sandbox/runner.py`
- UI flow: `frontend/app.js`
- API logic: `app/services/chat_service.py`

## Batch evaluation
Run the sample batch script only after the backend is up:

```bash
python scripts/batch_eval.py
```
