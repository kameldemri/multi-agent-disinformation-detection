# Phase 02 — Agent 1 Service

Branch: `phase/02-agent1-service`

Depends on: phase 01

Status: planned

## Goal

Expose Agent 1 as an internal HTTP service using the frozen contract.

## Scope

- Add FastAPI app for Agent 1.
- Endpoint: `POST /predict`.
- Input: article `text` and optional `title`.
- Output: `fake_news_probability` and `predicted_label`.
- Add tests using the tracked model bundle.
- Add `/health`.

## Acceptance Criteria

- Service runs independently.
- No training command needed.
- Sample article returns a valid probability in `[0, 1]`.
- Invalid empty text returns `422`.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/test_agent1_service.py
```

## Next Step

Phase 03: Agent 2 service.
