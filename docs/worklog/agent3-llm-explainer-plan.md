# Agent 3 LLM Explainer — Plan

Branch: `feat/agent3-llm-explainer`

Depends on: `feat/agent3-policy`

Status: merged into `main` at `3ccc516`; pushed.

## Goal

Add optional LLM explanation generation without changing deterministic decisions.

## Scope

- Added `LLM_MODE=offline|api`.
- Added OpenAI-compatible chat completion client using `httpx`.
- Added strict prompt and structured JSON parsing.
- LLM returns only `reason`.
- Deterministic core still owns `decision` and `final_risk_probability`.
- Added fallback path for offline mode, missing key, malformed JSON, and HTTP errors.

## Files Added

- `services/agent3_service/llm.py`
- `tests/test_agent3_llm.py`

## Files Changed

- `services/agent3_service/app.py`
- `requirements.txt`

## Dependency Changes

```text
httpx>=0.27,<1.0
```

## Environment Variables

- `LLM_MODE`, default `offline`.
- `LLM_API_KEY`.
- `LLM_BASE_URL`, default `https://api.openai.com/v1`.
- `LLM_MODEL`, default `gpt-4o-mini`.
- `LLM_TIMEOUT_SECONDS`, default `10`.
- `LLM_MAX_TOKENS`, default `160`.

## Acceptance Criteria

- Offline mode works without API key: passed.
- API mode can be enabled by env vars: passed with fake client.
- LLM cannot change `decision` or `final_risk_probability`: passed.
- Malformed LLM output falls back cleanly: passed.
- Secrets never written to repo or logs: passed.

## Commands Run

```bash
.venv/bin/python -m compileall -q services tests
.venv/bin/python -m pip check
.venv/bin/python -m pip install --dry-run -r requirements.txt
PYTHONPATH=. .venv/bin/python tests/test_agent3_llm.py
LLM_MODE=offline PYTHONPATH=. .venv/bin/python -m uvicorn services.agent3_service.app:app --host 127.0.0.1 --port 8014
```

## Verification Results

Offline HTTP output:

```json
{"user_id":"account_001","final_risk_probability":0.93,"decision":"BAN","reason":"Both behavioral and content evidence exceed their configured thresholds.","evidence":{"fake_news_probability":0.87,"bot_probability":0.93},"llm_status":"disabled"}
```

Path checks passed:

- `disabled` for offline mode.
- `used` for valid API response.
- `fallback` for missing key.
- Malformed JSON rejected.

A non-blocking `StarletteDeprecationWarning` appeared from `TestClient`; service behavior passed.

## Result

Agent 3 can optionally use an LLM for explanation while keeping enforcement decisions deterministic.

## Next Step

User confirms this branch; then create `feat/coordinator-api`.
