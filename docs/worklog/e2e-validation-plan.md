# End-to-End Validation — Plan

Branch: `test/agent3-e2e-validation`

Depends on: phase 07

Status: planned

## Goal

Validate the complete system from social media input to final decision.

## Scope

- Run decision matrix through public API.
- Run article-only and account-only smoke flows if supported.
- Verify Docker logs and healthchecks.
- Update final current-state document.

## Acceptance Criteria

- BAN and MODERATION scenarios correct.
- Boundary cases match thresholds.
- Invalid inputs return `422`.
- README/worklog state reflects final implementation.

## Tests

```bash
PYTHONPATH=. python -m pytest
docker compose up --build
```

## Next Step

User confirmation, then merge implementation branches in order into `main`.
