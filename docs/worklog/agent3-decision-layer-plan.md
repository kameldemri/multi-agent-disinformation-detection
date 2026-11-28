# Agent 3 Decision Layer — Planning

Branch: `plan/agent3-decision-layer`

Status: merged into `main` at `3ccc516`; pushed.

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
- `docs/worklog/agent3-decision-layer-plan.md`
- `docs/worklog/agent3-contracts.md`
- `docs/worklog/agent1-service-plan.md`
- `docs/worklog/agent2-service-plan.md`
- `docs/worklog/agent3-policy-plan.md`
- `docs/worklog/agent3-llm-explainer-plan.md`
- `docs/worklog/coordinator-api-plan.md`
- `docs/worklog/docker-runtime-plan.md`
- `docs/worklog/e2e-validation-plan.md`

## Commands Run

```bash
git status -sb
mkdir -p docs/worklog
```

## Result

Planning artifacts and frozen contracts created on one planning branch. No application code changed.

## Next Step

User confirms the plan; then create `feat/agent1-service`.
