# Docker Runtime — Plan

Branch: `infra/docker-runtime`

Depends on: phase 06

Status: merged into `main` at `3ccc516`; pushed

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
- Only coordinator is published to the host.

## Files Added

- `Dockerfile`
- `.dockerignore`
- `docker-compose.yml`
- `.env.example`
- `requirements-runtime.txt`

## Verification Results

Commands run on `infra/docker-runtime`:

```bash
docker compose config --quiet
docker compose build
docker compose up -d
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/analyze ...
docker compose logs --tail=20
docker compose down
```

Results:

- Compose config validated.
- Image built successfully with Python 3.12 slim and CPU torch `2.5.1+cpu`.
- Four containers started from the same Dockerfile.
- All four containers reported healthy.
- Only coordinator published `0.0.0.0:8000->8000/tcp`.
- `GET /health` returned `{"status":"ok"}`.
- `POST /analyze` returned agent 1, agent 2, and agent 3 outputs.
- Sample decision was `MODERATION` with risk `0.45518806499284004`.
- Stack was shut down cleanly after verification.

## Next Step

`test/agent3-e2e-validation`.
