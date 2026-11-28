# Agent 2 Service — Plan

Branch: `feat/agent2-service`

Depends on: `feat/agent1-service`

Status: done locally; awaiting user confirmation.

## Goal

Expose Agent 2 as an internal HTTP service using the frozen single-account contract.

## Scope

- Added FastAPI app for Agent 2.
- Endpoint: `POST /predict`.
- Input: one account object with `user_id` and `tweets`.
- Output: `user_id`, `bot_probability`, `predicted_label`.
- Added `/health`.
- Added model lazy-loading so the checkpoint is loaded once per service process.

## Files Added

- `services/agent2_service/__init__.py`
- `services/agent2_service/app.py`
- `tests/test_agent2_service.py`

## Environment Variables

- `BOT_THRESHOLD`, default `0.5`.
- `PORT`, default `8002`.

## Acceptance Criteria

- Service runs independently: passed.
- No training or preprocessing command needed: passed.
- Sample account returns valid bot probability in `[0, 1]`: passed.
- Fewer than 3 tweets returns `422`: passed.
- Timezone-naive timestamps return `422`: passed.
- Empty `user_id` returns `422`: passed.

## Commands Run

```bash
.venv/bin/python -m compileall -q services tests
PYTHONPATH=. .venv/bin/python tests/test_agent2_service.py
PYTHONPATH=. .venv/bin/python -m uvicorn services.agent2_service.app:app --host 127.0.0.1 --port 8012
```

## Verification Results

Test script output:

```json
{"user_id": "account_001", "bot_probability": 0.1048131138086319, "predicted_label": 0}
```

Live health:

```json
{"status":"ok"}
```

Live prediction:

```json
{"user_id":"account_001","bot_probability":0.1048131138086319,"predicted_label":0}
```

A non-blocking `StarletteDeprecationWarning` appeared from `TestClient`; service behavior passed.

## Result

Agent 2 is ready to be called as an internal HTTP service.

## Next Step

User confirms this branch; then create `feat/agent3-policy`.
