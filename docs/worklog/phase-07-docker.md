# Phase 07 — Docker

Branch: `phase/07-docker`

Depends on: phase 06

Status: planned

## Goal

Make the whole stack run anywhere with Docker Compose.

## Scope

- One image with all Python dependencies and tracked artifacts.
- Four containers: coordinator, agent1, agent2, agent3.
- Internal network.
- Only coordinator published to host.
- Env vars for thresholds and LLM config.
- Healthchecks.
- CPU torch setup.

## Acceptance Criteria

- `docker compose up --build` starts cleanly.
- `curl localhost:8000/health` works.
- Coordinator can reach all internal services.
- No API key required for default mode.

## Tests

```bash
docker compose up --build
curl http://localhost:8000/health
```

## Next Step

Phase 08: end-to-end validation.
