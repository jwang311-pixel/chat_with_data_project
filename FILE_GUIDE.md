# File guide

This project is split so future changes are easy to find.

## Core files
- `app/main.py` - app startup, static frontend, health check
- `app/routers/api.py` - HTTP endpoints
- `app/schemas.py` - request and response shapes

## Data loading
- `app/utils/csv_utils.py` - CSV reader that handles the extra title row found in some sample files
- `app/services/data_service.py` - dataset discovery, upload handling, and dataframe metadata

## LLM and prompts
- `app/services/openrouter_client.py` - OpenRouter request wrapper
- `app/services/prompt_service.py` - builds the prompt messages
- `app/prompts/system.txt` - base system instruction
- `app/prompts/*.txt` - one file per prompt mode
- `config/models.json` - the four model choices shown in the UI
- `config/prompt_modes.json` - the five prompt modes shown in the UI

## Execution sandbox
- `app/services/sandbox_service.py` - backend wrapper around the runner process
- `sandbox/runner.py` - executes generated code against `df`

## Frontend
- `frontend/index.html` - page structure
- `frontend/app.js` - UI logic and API calls
- `frontend/styles.css` - styling

## Experiment files
- `evaluation/questions/` - the provided question sets
- `scripts/batch_eval.py` - starter script for running a few questions through the API

## Best places to edit later
- Prompt behavior: `app/prompts/`
- Model list: `config/models.json`
- Execution safety/time limits: `sandbox/runner.py`
- UI flow: `frontend/app.js`
- API business logic: `app/services/chat_service.py`
