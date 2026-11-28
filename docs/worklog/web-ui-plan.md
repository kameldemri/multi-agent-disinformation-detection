# Web UI — Plan

Branch: `feat/web-ui`

Depends on: `infra/local-llm` at `1a621a9`

Status: done locally; not pushed

## Goal

Add a functional browser UI for submitting evidence and viewing all three agent outputs.

## Scope

- Serve the UI from the existing coordinator container on port `8000`.
- Keep the UI dependency-free: plain HTML, CSS, and JavaScript only.
- Call the existing public `POST /analyze` endpoint from the same origin.
- Show decisions, risk scores, probabilities, labels, explanation, and `llm_status`.
- Keep the default stack free and offline; the UI works with offline, BYO key, or local LLM modes.

## Design

- `web/index.html` contains the form and result area.
- `web/styles.css` provides responsive styling without external assets.
- `web/app.js` validates input, builds timestamps, calls `/analyze`, and renders results.
- `services/coordinator_service/app.py` mounts `/static` and serves `/` from `web/index.html`.
- No new Python dependency is required; FastAPI/Starlette static support is already available.

## Acceptance Criteria

- `GET /` returns the browser UI.
- `/static/app.js` and `/static/styles.css` are served correctly.
- The form can submit text, optional title, user ID, and at least three timestamps.
- Invalid input is shown clearly in the UI.
- Results show `MODERATION`/`BAN`, risk values, agent labels, explanation, raw JSON, and LLM status.
- Existing API behavior and E2E validation remain intact.

## Verification Results

Commands run on `feat/web-ui`:

```bash
PYTHONPATH=. python -m compileall -q services tests
PYTHONPATH=. python tests/test_agent1_service.py
PYTHONPATH=. python tests/test_agent2_service.py
PYTHONPATH=. python tests/test_agent3_policy.py
PYTHONPATH=. python tests/test_agent3_llm.py
PYTHONPATH=. python tests/test_coordinator.py
docker compose --profile llm up -d --build coordinator
```

Results:

- `compileall` passed.
- All service test scripts passed locally.
- Coordinator tests now cover `/`, `/static/app.js`, and `/static/styles.css`.
- `GET /` returned `200 text/html`.
- `GET /static/app.js` returned `200 text/javascript`.
- `GET /static/styles.css` returned `200 text/css`.
- E2E validation with UI checks passed for `MODERATION` at thresholds `0.9`/`0.9`.
- E2E validation with UI checks passed for `BAN` at thresholds `0.05`/`0.05`.
- Both E2E decisions returned `llm_status="used"` with local LLM enabled.
- Stack was restored healthy for direct use.

## Next Step

User confirmation, then commit this branch and merge/push when requested.
