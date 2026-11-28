# Phase 00 — Planning

Branch: `phase/00-planning`

Status: done on this branch; pending user confirmation before merge.

## Goal

Write enough plan and context so work can pause or move to another agent without losing decisions.

## Decisions Preserved

- One branch per phase.
- Informal worklog under `docs/worklog/`.
- Worklog committed.
- Per-phase Markdown files plus a living current-state file.
- Stop after each phase for user confirmation.
- Three-agent architecture with coordinator.
- Agent 3 per account.
- Deterministic BAN/MODERATION policy.
- `final_risk_probability = max(bot_probability, fake_news_probability)`.
- Invalid/missing evidence returns `422`.
- LLM explains only; it does not decide.
- Optional API LLM with offline fallback.
- One image, four containers.

## Files Created

- `docs/worklog/README.md`
- `docs/worklog/current-state.md`
- `docs/worklog/phase-00-planning.md`
- `docs/worklog/phase-01-contracts.md`
- `docs/worklog/phase-02-agent1-service.md`
- `docs/worklog/phase-03-agent2-service.md`
- `docs/worklog/phase-04-agent3-policy.md`
- `docs/worklog/phase-05-agent3-llm.md`
- `docs/worklog/phase-06-coordinator.md`
- `docs/worklog/phase-07-docker.md`
- `docs/worklog/phase-08-e2e-validation.md`

## Commands Run

```bash
git status -sb
git checkout -b phase/00-planning
mkdir -p docs/worklog
```

## Result

Planning artifacts created. No application code changed.

## Next Step

Create `phase/01-contracts` from this branch and complete Phase 1.
