# Phase 06 — Coordinator

Branch: `phase/06-coordinator`

Depends on: phase 05

Status: planned

## Goal

Expose one public API that orchestrates all agents.

## Scope

- Add FastAPI coordinator service.
- Endpoint: `POST /analyze`.
- Input includes article text/title and one account.
- Calls Agent 1, Agent 2, then Agent 3.
- Returns all agent outputs plus final Agent 3 result.
- Add `/health`.

## Acceptance Criteria

- One request produces a final decision.
- Internal service errors are handled cleanly.
- Invalid evidence is propagated as `422`.
- Public API does not expose internal agent URLs in responses.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/test_coordinator.py
```

## Next Step

Phase 07: Dockerization.
