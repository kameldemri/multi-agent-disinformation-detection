# Worklog Index

This folder is an informal working log, not official product documentation.

Its purpose is to preserve enough context for another agent, or for us after context loss, to continue the project safely.

## Rules

- Planning and contracts live on one planning branch.
- A separate branch is created only when a phase changes application code or infrastructure.
- Do not merge a phase branch into `main` until the user confirms it.
- Do not push documentation-only verification work unless the user explicitly asks.
- After each finished implementation phase, update:
  - `current-state.md`
  - this `README.md`
  - that phase's plan file
- Commit every worklog update with the phase work.
- If remote push fails because of GitHub credentials, ask the user to push manually.

## Current Base

- Base branch before planning: `main`
- Base commit: `cf8d8259ee52b07c0f1b6fa210dd0517b5f4a6b3`
- Planning branch: `plan/agent3-decision-layer`
- Current active branch: `infra/docker-runtime`
- Planning is committed locally but not pushed.

## Phase Status

| Phase | Branch | Status |
| --- | --- | --- |
| Planning + contracts | `plan/agent3-decision-layer` | done locally; not pushed |
| Agent 1 service | `feat/agent1-service` | done locally; awaiting confirmation |
| Agent 2 service | `feat/agent2-service` | done locally; awaiting confirmation |
| Agent 3 policy | `feat/agent3-policy` | done locally; awaiting confirmation |
| Agent 3 LLM explainer | `feat/agent3-llm-explainer` | done locally; awaiting confirmation |
| Coordinator API | `feat/coordinator-api` | done locally; awaiting confirmation |
| Docker runtime | `infra/docker-runtime` | done locally; awaiting confirmation |
| End-to-end validation | `test/agent3-e2e-validation` | planned |

## Core Files

- `current-state.md`: living snapshot of decisions, state, contracts, and next action.
- `agent3-decision-layer-plan.md`: complete implementation plan.
- `agent3-contracts.md`: frozen contracts from planning.
- `agent1-service-plan.md`, `agent2-service-plan.md`, `agent3-policy-plan.md`, `agent3-llm-explainer-plan.md`, `coordinator-api-plan.md`, `docker-runtime-plan.md`, `e2e-validation-plan.md`: per-phase plans.

## Context Recovery Commands

```bash
git status -sb
git branch --show-current
git log --oneline -8 --decorate
cat docs/worklog/current-state.md
cat docs/worklog/README.md
```
