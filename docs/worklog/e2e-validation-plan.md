# End-to-End Validation — Plan

Branch: `test/agent3-e2e-validation`

Depends on: phase 07

Status: merged into `main` at `3ccc516`; pushed

## Goal

Validate the complete system from social media input to final decision.

## Scope

- Run BAN and MODERATION scenarios through the public API.
- Verify threshold-driven labels and risk.
- Verify invalid inputs return `422`.
- Verify Docker logs and healthchecks.
- Update final current-state document and README.

## Acceptance Criteria

- BAN and MODERATION scenarios correct.
- Boundary behavior matches configured thresholds.
- Invalid inputs return `422`.
- README/worklog state reflects final implementation.

## Files Added

- `tests/e2e_validation.py`
- `scripts/e2e_validation.sh`

## Verification Results

Commands run on `test/agent3-e2e-validation`:

```bash
PYTHONPATH=. python -m compileall -q services tests
PYTHONPATH=. python tests/test_agent1_service.py
PYTHONPATH=. python tests/test_agent2_service.py
PYTHONPATH=. python tests/test_agent3_policy.py
PYTHONPATH=. python tests/test_agent3_llm.py
PYTHONPATH=. python tests/test_coordinator.py
PYTHON_BIN=python bash scripts/e2e_validation.sh
```

Results:

- `compileall` passed.
- All existing service test scripts passed locally.
- High-threshold Compose scenario returned `MODERATION`.
- Low-threshold Compose scenario returned `BAN`.
- Threshold-driven `predicted_label` values matched configuration.
- `final_risk_probability` matched `max(fake_news_probability, bot_probability)`.
- Invalid text, missing text, empty user, short account, and naive timestamp returned `422`.
- Response shape, evidence mapping, user id, reason, and `llm_status` were checked.
- Internal agent URLs were not exposed.
- Docker healthchecks and logs were checked in both scenarios.
- Both Compose scenarios shut down cleanly.

Article-only and account-only public flows are not supported; `/analyze` requires both text and account evidence.

## Next Step

User confirmation, then merge implementation branches in order into `main`.
