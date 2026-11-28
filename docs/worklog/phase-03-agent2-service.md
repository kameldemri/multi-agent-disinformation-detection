# Phase 03 — Agent 2 Service

Branch: `phase/03-agent2-service`

Depends on: phase 02

Status: planned

## Goal

Expose Agent 2 as an internal HTTP service using the frozen single-account contract.

## Scope

- Add FastAPI app for Agent 2.
- Endpoint: `POST /predict`.
- Input: one account object with `user_id` and `tweets`.
- Output: `user_id`, `bot_probability`, `predicted_label`.
- Add tests using tracked checkpoint and scaler.
- Add `/health`.

## Acceptance Criteria

- Service runs independently.
- No training or preprocessing command needed.
- Sample account returns valid bot probability in `[0, 1]`.
- Fewer than 3 tweets returns `422`.
- Timezone-naive timestamps return `422`.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/test_agent2_service.py
```

## Next Step

Phase 04: Agent 3 deterministic policy.
