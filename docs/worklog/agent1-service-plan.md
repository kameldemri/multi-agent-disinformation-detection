# Agent 1 Service — Plan

Branch: `feat/agent1-service`

Depends on: `plan/agent3-decision-layer`

Status: merged into `main` at `3ccc516`; pushed.

## Goal

Expose Agent 1 as an internal HTTP service using the frozen contract.

## Scope

- Added FastAPI app for Agent 1.
- Endpoint: `POST /predict`.
- Input: article `text` and optional `title`.
- Output: `fake_news_probability` and `predicted_label`.
- Added `/health`.
- Added runtime dependencies.

## Files Added

- `services/__init__.py`
- `services/agent1_service/__init__.py`
- `services/agent1_service/app.py`
- `tests/test_agent1_service.py`

## Files Changed

- `requirements.txt`

## Dependency Changes

```text
fastapi>=0.115,<1.0
uvicorn>=0.30,<1.0
```

## Environment Variables

- `FAKE_NEWS_THRESHOLD`, default `0.5`.
- `PORT`, default `8001`.

## Acceptance Criteria

- Service runs independently: passed.
- No training command needed: passed.
- Sample article returns a valid probability in `[0, 1]`: passed.
- Invalid empty text returns `422`: passed.
- Missing `text` returns `422`: passed.

## Commands Run

```bash
.venv/bin/python -m compileall -q services tests
.venv/bin/python -m pip check
.venv/bin/python -m pip install --dry-run -r requirements.txt
PYTHONPATH=. .venv/bin/python tests/test_agent1_service.py
PYTHONPATH=. .venv/bin/python -m uvicorn services.agent1_service.app:app --host 127.0.0.1 --port 8011
```

## Verification Results

Test script output:

```json
{"fake_news_probability": 0.45518806499284004, "predicted_label": 0}
```

Live health:

```json
{"status":"ok"}
```

Live prediction:

```json
{"fake_news_probability":0.45518806499284004,"predicted_label":0}
```

A non-blocking `StarletteDeprecationWarning` appeared from `TestClient`; service behavior passed.

## Result

Agent 1 is ready to be called as an internal HTTP service.

## Next Step

User confirms this branch; then create `feat/agent2-service`.
