# Phase 05 — Agent 3 LLM

Branch: `phase/05-agent3-llm`

Depends on: phase 04

Status: planned

## Goal

Add optional LLM explanation generation without changing deterministic decisions.

## Scope

- Add `LLM_MODE=offline|api`.
- Add OpenAI-compatible client using env vars.
- Strict prompt with policy and evidence.
- LLM returns only `reason`.
- Coordinator/Agent 3 fills final structured response.
- Fallback if no key, timeout, refusal, invalid JSON, or invalid decision text.

## Acceptance Criteria

- Offline mode works without API key.
- API mode can be enabled by env vars.
- LLM cannot change `decision` or `final_risk_probability`.
- Malformed LLM output falls back cleanly.
- Secrets never written to repo or logs.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/test_agent3_llm.py
```

## Next Step

Phase 06: coordinator service.
