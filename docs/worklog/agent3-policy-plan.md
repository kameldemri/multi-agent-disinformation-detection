# Agent 3 Policy — Plan

Branch: `feat/agent3-policy`

Depends on: phase 03

Status: planned

## Goal

Implement Agent 3 deterministic decision core before adding the LLM.

## Scope

- Add Pydantic input/output models.
- Validate probabilities in `[0, 1]`.
- Validate matching `user_id`.
- Apply BAN/MODERATION policy.
- Compute `final_risk_probability = max(...)`.
- Generate fallback deterministic `reason`.
- Add decision-matrix tests.

## Acceptance Criteria

- Bot above threshold returns BAN.
- Fake news above threshold returns BAN.
- Both below thresholds returns MODERATION.
- Missing/invalid evidence returns `422`.
- LLM not required.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/test_agent3_policy.py
```

## Next Step

`feat/agent3-llm-explainer`.
