# Worklog Index

This folder is an informal working log, not official product documentation.

Its purpose is to preserve enough context for another agent, or for us after context loss, to continue the project safely.

## Rules

- Each phase gets its own branch.
- Do not merge a phase branch into `main` until the user confirms it.
- After each finished phase, update:
  - `current-state.md`
  - this `README.md`
  - that phase's `phase-NN-*.md`
- Commit every worklog update with the phase work.
- If remote push fails because of GitHub credentials, ask the user to push manually.

## Current Base

- Base branch before planning: `main`
- Base commit: `cf8d8259ee52b07c0f1b6fa210dd0517b5f4a6b3`
- Planning branch: `phase/00-planning`

## Phase Status

| Phase | Branch | Status |
| --- | --- | --- |
| 00 Planning | `phase/00-planning` | in progress |
| 01 Contracts | `phase/01-contracts` | planned |
| 02 Agent 1 service | `phase/02-agent1-service` | planned |
| 03 Agent 2 service | `phase/03-agent2-service` | planned |
| 04 Agent 3 policy | `phase/04-agent3-policy` | planned |
| 05 Agent 3 LLM | `phase/05-agent3-llm` | planned |
| 06 Coordinator | `phase/06-coordinator` | planned |
| 07 Docker | `phase/07-docker` | planned |
| 08 End-to-end validation | `phase/08-e2e-validation` | planned |

## Core Files

- `current-state.md`: living snapshot of decisions, state, contracts, and next action.
- `phase-00-planning.md`: complete implementation plan.
- `phase-NN-*.md`: one file per phase.

## Context Recovery Commands

```bash
git status -sb
git branch --show-current
git log --oneline -8 --decorate
cat docs/worklog/current-state.md
cat docs/worklog/README.md
```
