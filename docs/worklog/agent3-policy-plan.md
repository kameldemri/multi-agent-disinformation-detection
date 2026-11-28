# Agent 3 Policy — Plan

Branch: `feat/agent3-policy`

Depends on: `feat/agent2-service`

Status: done locally; awaiting user confirmation.

## Goal

Implement Agent 3 deterministic decision core before adding the LLM.

## Scope

- Added Pydantic input/output models.
- Added validation for probabilities in `[0, 1]`.
- Added validation that top-level `user_id` matches `agent_2.user_id`.
- Added BAN/MODERATION policy.
- Added `final_risk_probability = max(...)`.
- Added deterministic fallback reason.
- Added FastAPI service with `/decide` and `/health`.

## Files Added

- `services/agent3_service/__init__.py`
- `services/agent3_service/schemas.py`
- `services/agent3_service/policy.py`
- `services/agent3_service/app.py`
- `tests/test_agent3_policy.py`

## Environment Variables

- `BOT_THRESHOLD`, default `0.5`.
- `FAKE_NEWS_THRESHOLD`, default `0.5`.
- `PORT`, default `8003`.

## Acceptance Criteria

- Bot above threshold returns BAN: passed.
- Fake news above threshold returns BAN: passed.
- Both below thresholds returns MODERATION: passed.
- Missing/invalid evidence returns `422`: passed.
- LLM not required: passed.

## Commands Run

```bash
.venv/bin/python -m compileall -q services tests
PYTHONPATH=. .venv/bin/python tests/test_agent3_policy.py
PYTHONPATH=. .venv/bin/python -m uvicorn services.agent3_service.app:app --host 127.0.0.1 --port 8013
```

## Verification Results

Decision matrix passed:

| Bot probability | Fake probability | Decision |
| ---: | ---: | --- |
| 0.95 | 0.10 | BAN |
| 0.90 | 0.90 | BAN |
| 0.10 | 0.95 | BAN |
| 0.10 | 0.10 | MODERATION |
| 0.49 | 0.49 | MODERATION |
| 0.50 | 0.10 | BAN |
| 0.10 | 0.50 | BAN |

Live decide output:

```json
{"user_id":"account_001","final_risk_probability":0.93,"decision":"BAN","reason":"Both behavioral and content evidence exceed their configured thresholds.","evidence":{"fake_news_probability":0.87,"bot_probability":0.93},"llm_status":"disabled"}
```

A non-blocking `StarletteDeprecationWarning` appeared from `TestClient`; service behavior passed.

## Result

Agent 3 policy is deterministic, validated, and ready for the optional LLM explanation layer.

## Next Step

User confirms this branch; then create `feat/agent3-llm-explainer`.
