# Coordinator API — Plan

Branch: `feat/coordinator-api`

Depends on: `feat/agent3-llm-explainer`

Status: merged into `main` at `3ccc516`; pushed.

## Goal

Expose one public API that orchestrates all agents.

## Scope

- Added FastAPI coordinator service.
- Added `POST /analyze`.
- Input includes article text/title and one account.
- Calls Agent 1, Agent 2, then Agent 3.
- Returns all agent outputs plus final Agent 3 result.
- Added `/health`.
- Added internal HTTP client with `422` propagation and `502`/`504` handling.

## Files Added

- `services/coordinator_service/__init__.py`
- `services/coordinator_service/schemas.py`
- `services/coordinator_service/client.py`
- `services/coordinator_service/app.py`
- `tests/test_coordinator.py`

## Environment Variables

- `AGENT1_URL`, default `http://agent1:8001`.
- `AGENT2_URL`, default `http://agent2:8002`.
- `AGENT3_URL`, default `http://agent3:8003`.
- `COORDINATOR_TIMEOUT_SECONDS`, default `30`.
- `PORT`, default `8000`.

## Acceptance Criteria

- One request produces a final decision: passed.
- Internal service errors are handled cleanly: passed.
- Invalid evidence is propagated as `422`: passed.
- Public API does not expose internal agent URLs in responses: passed.

## Commands Run

```bash
.venv/bin/python -m compileall -q services tests
PYTHONPATH=. .venv/bin/python tests/test_coordinator.py
```

Live four-service stack:

```bash
PYTHONPATH=. .venv/bin/python -m uvicorn services.agent1_service.app:app --host 127.0.0.1 --port 8111
PYTHONPATH=. .venv/bin/python -m uvicorn services.agent2_service.app:app --host 127.0.0.1 --port 8112
LLM_MODE=offline PYTHONPATH=. .venv/bin/python -m uvicorn services.agent3_service.app:app --host 127.0.0.1 --port 8113
AGENT1_URL=http://127.0.0.1:8111 AGENT2_URL=http://127.0.0.1:8112 AGENT3_URL=http://127.0.0.1:8113 PYTHONPATH=. .venv/bin/python -m uvicorn services.coordinator_service.app:app --host 127.0.0.1 --port 8114
```

## Verification Results

Live `/analyze` output:

```json
{"agent_1":{"fake_news_probability":0.45518806499284004,"predicted_label":0},"agent_2":{"user_id":"account_001","bot_probability":0.1048131138086319,"predicted_label":0},"agent_3":{"user_id":"account_001","final_risk_probability":0.45518806499284004,"decision":"MODERATION","reason":"Both content and behavioral risks are below their configured thresholds.","evidence":{"fake_news_probability":0.45518806499284004,"bot_probability":0.1048131138086319},"llm_status":"disabled"}}
```

A non-blocking `StarletteDeprecationWarning` appeared from `TestClient`; service behavior passed.

## Result

The public orchestration API is ready.

## Next Step

User confirms this branch; then create `infra/docker-runtime`.
