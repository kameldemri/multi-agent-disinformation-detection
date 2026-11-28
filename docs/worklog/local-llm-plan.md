# Local LLM — Plan

Branch: `infra/local-llm`

Depends on: final merged system at `3ccc516`

Status: done locally; not pushed

## Goal

Add an optional free local LLM explanation backend that requires no API key.

## Scope

- Keep `offline` mode as the default.
- Keep BYO API key mode available.
- Allow OpenAI-compatible local endpoints without an `Authorization` header.
- Add an optional Compose profile for a local model server.
- Avoid baking model weights into the application image.

## Design

- `LLM_ALLOW_NO_KEY=true` permits API mode with no key.
- Local model server runs only with `--profile llm`.
- Local model is pulled on first setup into a named Docker volume.
- Agent 3 still controls the decision and risk; LLM only writes `reason`.
- Failures fall back to the deterministic reason with `llm_status="fallback"`.

## Acceptance Criteria

- Default `docker compose up` stays unchanged and free.
- `docker compose --profile llm config` validates.
- Local no-key API mode passes with a fake client.
- No secret or API key is committed.
- Worklog and README document usage.

## Usage

```bash
cp .env.example .env
```

Set in `.env`:

```env
LLM_MODE=api
LLM_ALLOW_NO_KEY=true
LLM_BASE_URL=http://llm:11434/v1
LLM_MODEL=qwen2.5:0.5b-instruct
LLM_LOCAL_MODEL=qwen2.5:0.5b-instruct
```

Start:

```bash
docker compose --profile llm up -d --build
```

## Verification Results

Commands run on `infra/local-llm`:

```bash
PYTHONPATH=. python -m compileall -q services tests
PYTHONPATH=. python tests/test_agent1_service.py
PYTHONPATH=. python tests/test_agent2_service.py
PYTHONPATH=. python tests/test_agent3_policy.py
PYTHONPATH=. python tests/test_agent3_llm.py
PYTHONPATH=. python tests/test_coordinator.py
docker compose config --quiet
docker compose --profile llm config --quiet
```

Results:

- `compileall` passed.
- All existing service test scripts passed locally.
- Local no-key API mode passed with a fake client.
- No `Authorization` header is sent when `LLM_ALLOW_NO_KEY=true` and no key is configured.
- Missing key still falls back when `LLM_ALLOW_NO_KEY=false`.
- Default Compose config validated.
- LLM-profile Compose config validated.
- No real model was downloaded and no real API call was made during verification.

## Next Step

User confirmation, then merge `infra/local-llm` into `main`.
