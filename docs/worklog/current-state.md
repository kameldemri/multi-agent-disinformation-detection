# Current State

Last updated: 2026-11-28

## Objective

Build a final system with three distinct agent responsibilities:

- Agent 1: content-level fake-news detection.
- Agent 2: account-level bot detection from temporal tweet behavior.
- Agent 3: final decision layer that combines both evidence signals and applies a controlled policy.

Then expose everything through a coordination layer and Dockerize it so it runs anywhere.

## Git State

- Base `main` commit: `cf8d8259ee52b07c0f1b6fa210dd0517b5f4a6b3`.
- Base `main` message: `fix(deps): declare joblib and torch requirements`.
- Planning branch: `plan/agent3-decision-layer`.
- Current branch: `infra/local-llm`.
- Final merged/pushed `main` commit: `3ccc516`.
- Planning docs and all implementation phases through E2E validation are merged into `main` and pushed.
- Documentation-only work is folded into the planning branch; no throwaway phase branch.
- Implementation phases get professional branches such as `feat/agent1-service`.
- Do not merge a new phase into `main` until the user confirms.

## Completed Baseline

- PR #8 and PR #9 are merged.
- `requirements.txt` includes `joblib>=1.4,<2.0` and `torch>=2.2,<3.0`.
- Agent 1 and Agent 2 run locally without training or raw datasets.
- Agent 1 artifact: `agents/agent1_naive_bayes/models/nb_fake_news.joblib`.
- Agent 2 artifacts:
  - `agents/agent2_rnn/artifacts/best_bigru_safe.pt`
  - `data/processed/agent2/temporal_scaler.npz`
- Agent 2 tests passed on merged `main`.
- Python packages are still required; inference is not dependency-free.

## User Decisions

- Each phase must have its own branch.
- Worklog Markdown files live in `docs/worklog/`.
- Worklog files are committed.
- Worklog structure: `README.md` index + `current-state.md` + one file per phase.
- Stop after each phase branch and wait for user confirmation.
- Agent 3 is per account.
- `final_risk_probability = max(bot_probability, fake_news_probability)`.
- Missing/invalid agent evidence returns `422`.
- Agent 3 decision policy is deterministic.
- LLM only explains the deterministic decision.
- LLM cannot change `decision` or `final_risk_probability`.
- LLM backend is optional API mode with offline fallback.
- No shared/public API key will be committed; users bring their own key or use the free local model.
- Local LLM is an optional Compose profile, not part of the default stack.
- Docker structure: one image, one service per agent plus one coordinator service.
- Docker networking: internal network; only coordinator published to host.
- Agent 2 input stays timestamp-only for now; no tweet text decision until further notice.

## Agent 1 Current API

File: `agents/agent1_naive_bayes/src/predict.py`

Raw functions:

```python
score_text(text: str, title: str = "") -> float
score_texts(texts: list[str], titles: list[str] | None = None) -> list[float]
explain_text(text: str, title: str = "", n: int = 8) -> dict
```

Meaning:

- `0.0` = surely true.
- `1.0` = surely fake.
- `0.5` = undecided / no known words.
- CLI currently labels `FAKE` if `p >= 0.5`, else `TRUE`.

Important limitation:

- The model was trained on English news articles, not tweets.

Normalized service contract to freeze:

```json
{
  "fake_news_probability": 0.455,
  "predicted_label": 0
}
```

`predicted_label` will be computed by the Agent 1 service using `FAKE_NEWS_THRESHOLD`.

## Agent 2 Current API

File: `agents/agent2_rnn/src/predict.py`

Main function:

```python
predict_accounts(
    accounts,
    model=None,
    device=None,
    scaler_path=SCALER_PATH,
    batch_size=16,
    threshold=0.5,
) -> list[dict]
```

Raw input:

```json
[
  {
    "user_id": "account_001",
    "tweets": [
      {"tweet_id": "101", "created_at": "2026-11-20T10:30:00Z"},
      {"tweet_id": "102", "created_at": "2026-11-20T10:35:00Z"},
      {"tweet_id": "103", "created_at": "2026-11-20T11:00:00Z"}
    ]
  }
]
```

Constraints:

- `accounts` must be a non-empty list.
- Each account must have a valid `user_id`.
- Each account needs at least 3 tweets.
- Each tweet needs a valid `tweet_id` and `created_at`.
- Timestamps must include timezone information.
- Supported formats include Twitter format and ISO 8601 with timezone.
- Tweet text is not used by Agent 2.

Raw output:

```json
[
  {
    "user_id": "account_001",
    "bot_probability": 0.1048,
    "predicted_label": 0
  }
]
```

Normalized Agent 2 service contract is one account object per request:

```json
{
  "user_id": "account_001",
  "bot_probability": 0.1048,
  "predicted_label": 0
}
```

## Agent 3 Planned Contract

Input:

```json
{
  "user_id": "account_001",
  "agent_1": {
    "fake_news_probability": 0.87,
    "predicted_label": 1
  },
  "agent_2": {
    "user_id": "account_001",
    "bot_probability": 0.93,
    "predicted_label": 1
  }
}
```

Validation:

- `user_id` must match `agent_2.user_id`.
- Probabilities must be floats in `[0, 1]`.
- Missing or invalid evidence returns `422`.

Deterministic policy:

```python
if bot_probability >= BOT_THRESHOLD:
    decision = "BAN"
elif fake_news_probability >= FAKE_NEWS_THRESHOLD:
    decision = "BAN"
else:
    decision = "MODERATION"
```

Risk:

```python
final_risk_probability = max(bot_probability, fake_news_probability)
```

Output:

```json
{
  "user_id": "account_001",
  "final_risk_probability": 0.93,
  "decision": "BAN",
  "reason": "The account shows a high probability of bot-like behavior.",
  "evidence": {
    "fake_news_probability": 0.87,
    "bot_probability": 0.93
  },
  "llm_status": "used"
}
```

`llm_status` values:

- `used`: LLM explanation was used.
- `fallback`: deterministic explanation was used.
- `disabled`: no LLM configured.

## Planned Service Architecture

- `coordinator`: public API, calls all agents.
- `agent1`: wraps fake-news model.
- `agent2`: wraps BiGRU bot model.
- `agent3`: deterministic policy + optional LLM explanation.

Docker:

- One image, four containers.
- Internal Docker network.
- Only coordinator exposed to host.
- Secrets via environment variables only.

## Environment Variables Planned

- `BOT_THRESHOLD` default `0.5`.
- `FAKE_NEWS_THRESHOLD` default `0.5`.
- `LLM_MODE` values `offline` or `api`.
- `LLM_API_KEY`.
- `LLM_BASE_URL` for OpenAI-compatible providers.
- `LLM_MODEL`.
- `LLM_TIMEOUT_SECONDS`.
- `LLM_MAX_TOKENS`.
- `AGENT1_URL`, default `http://agent1:8001`.
- `AGENT2_URL`, default `http://agent2:8002`.
- `AGENT3_URL`, default `http://agent3:8003`.
- `COORDINATOR_TIMEOUT_SECONDS`, default `30`.
- `COORDINATOR_PORT`, default `8000`.
- `LOG_LEVEL`.

## Planning Verification Results

Commands run on `plan/agent3-decision-layer`:

```bash
PYTHONPATH=. .venv/bin/python -c "from agents.agent1_naive_bayes.src.predict import score_text; print(score_text('The election was rigged and officials hid the truth'))"
PYTHONPATH=. .venv/bin/python -c "from agents.agent2_rnn.src.predict import predict_accounts; print(predict_accounts([{'user_id':'account_001','tweets':[{'tweet_id':'101','created_at':'2026-11-20T10:30:00Z'},{'tweet_id':'102','created_at':'2026-11-20T10:35:00Z'},{'tweet_id':'103','created_at':'2026-11-20T11:00:00Z'}]}]))"
```

Results:

- Agent 1 raw score: `0.45518806499284004`.
- Agent 2 raw output:
  ```python
  [{'user_id': 'account_001', 'bot_probability': 0.1048131138086319, 'predicted_label': 0}]
  ```

No application code changed during planning.

## Agent 1 Service Result

New files:

- `services/agent1_service/app.py`
- `services/agent1_service/__init__.py`
- `services/__init__.py`
- `tests/test_agent1_service.py`

Dependency changes:

- Added `fastapi>=0.115,<1.0`.
- Added `uvicorn>=0.30,<1.0`.

Verified:

- `compileall` passed.
- `pip check` passed.
- `pip install --dry-run -r requirements.txt` passed.
- Test script passed with:
  ```json
  {"fake_news_probability": 0.45518806499284004, "predicted_label": 0}
  ```
- Live uvicorn check passed:
  - `GET /health` returned `{"status":"ok"}`.
  - `POST /predict` returned the expected JSON.

## Agent 2 Service Result

New files:

- `services/agent2_service/app.py`
- `services/agent2_service/__init__.py`
- `tests/test_agent2_service.py`

Verified:

- `compileall` passed.
- Test script passed with:
  ```json
  {"user_id": "account_001", "bot_probability": 0.1048131138086319, "predicted_label": 0}
  ```
- Live uvicorn check passed:
  - `GET /health` returned `{"status":"ok"}`.
  - `POST /predict` returned the expected JSON.
- Fewer than 3 tweets returns `422`.
- Timezone-naive timestamps return `422`.
- Empty `user_id` returns `422`.

## Agent 3 Policy Result

New files:

- `services/agent3_service/__init__.py`
- `services/agent3_service/schemas.py`
- `services/agent3_service/policy.py`
- `services/agent3_service/app.py`
- `tests/test_agent3_policy.py`

Verified:

- `compileall` passed.
- Decision matrix passed for all 7 cases.
- `final_risk_probability = max(bot_probability, fake_news_probability)` verified.
- `GET /health` returned `{"status":"ok"}`.
- `POST /decide` returned a structured BAN response.
- Mismatched `user_id` returns `422`.
- Probability outside `[0, 1]` returns `422`.
- `llm_status` is `disabled` in this phase.

## Agent 3 LLM Explainer Result

New file:

- `services/agent3_service/llm.py`

Changed files:

- `services/agent3_service/app.py`
- `requirements.txt`
- `tests/test_agent3_llm.py`

Verified:

- Offline mode returns deterministic reason with `llm_status="disabled"`.
- API success with a fake client returns LLM reason with `llm_status="used"`.
- Missing API key falls back to deterministic reason with `llm_status="fallback"`.
- Malformed JSON is rejected.
- `GET /health` and `POST /decide` passed live uvicorn check in offline mode.
- No real API call was made during tests.
- Decision and risk remain deterministic and unchanged by the LLM.

## Coordinator API Result

New files:

- `services/coordinator_service/__init__.py`
- `services/coordinator_service/schemas.py`
- `services/coordinator_service/client.py`
- `services/coordinator_service/app.py`
- `tests/test_coordinator.py`

Verified:

- `compileall` passed.
- Fake-client test passed for call order and response shape.
- Agent 2 `422` is propagated by the coordinator.
- Empty `text` returns `422`.
- Live four-service stack passed:
  - Coordinator `/health` returned `{"status":"ok"}`.
  - `/analyze` returned agent 1, agent 2, and agent 3 outputs.
- Internal agent URLs are not included in responses.

## Docker Runtime Result

New files:

- `Dockerfile`
- `.dockerignore`
- `docker-compose.yml`
- `.env.example`
- `requirements-runtime.txt`

Verified:

- `docker compose config --quiet` passed.
- `docker compose build` succeeded.
- Image built with Python 3.12 slim and CPU torch `2.5.1+cpu`.
- Four containers started from the same Dockerfile.
- All four containers reported healthy.
- Only coordinator published `0.0.0.0:8000->8000/tcp`.
- `GET /health` returned `{"status":"ok"}`.
- `POST /analyze` returned agent 1, agent 2, and agent 3 outputs.
- Sample decision was `MODERATION` with risk `0.45518806499284004`.
- `docker compose down` shut everything down cleanly.

## End-to-End Validation Result

New files:

- `tests/e2e_validation.py`
- `scripts/e2e_validation.sh`

Verified:

- `compileall` passed for `services` and `tests`.
- All existing service test scripts passed locally.
- Docker Compose scenario with `BOT_THRESHOLD=0.9` and `FAKE_NEWS_THRESHOLD=0.9` returned `MODERATION`.
- Docker Compose scenario with `BOT_THRESHOLD=0.05` and `FAKE_NEWS_THRESHOLD=0.05` returned `BAN`.
- Threshold-driven `predicted_label` values matched configured thresholds.
- `final_risk_probability` matched `max(fake_news_probability, bot_probability)`.
- Invalid cases returned `422`:
  - blank text
  - missing text
  - empty `user_id`
  - fewer than 3 tweets
  - timezone-naive timestamp
- Response shape, evidence mapping, user id, reason, and `llm_status` were checked.
- Internal agent URLs were not exposed in public responses.
- Docker healthchecks and logs were checked in both scenarios.
- Both scenarios shut down cleanly.

Article-only and account-only public flows are not supported; `/analyze` requires both text and account evidence.

## Local LLM Result

Changed files:

- `services/agent3_service/llm.py`
- `tests/test_agent3_llm.py`
- `docker-compose.yml`
- `.env.example`
- `README.md`

New file:

- `docs/worklog/local-llm-plan.md`

Verified:

- `compileall` passed for `services` and `tests`.
- All existing service test scripts passed locally.
- Local no-key API mode passed with a fake client.
- No `Authorization` header is sent when `LLM_ALLOW_NO_KEY=true` and no key is configured.
- Missing key still falls back when `LLM_ALLOW_NO_KEY=false`.
- Default `docker compose config --quiet` passed.
- `docker compose --profile llm config --quiet` passed.
- Real `qwen2.5:0.5b-instruct` model pulled successfully into `ollama_models`.
- Direct OpenAI-compatible request from Agent 3 to Ollama succeeded.
- Initial `/analyze` real-model call fell back because the model returned plain text instead of JSON.
- `services/agent3_service/llm.py` now accepts JSON or plain-text reason output; `tests/test_agent3_llm.py` covers the plain-text case.
- Final real-model `/analyze` call returned `llm_status="used"` with reason: `The decision is to MODERATE. The evidence shows that both content and behavioral risks are below their configured thresholds.`
- Full E2E validation was repeated with the local LLM profile enabled.
- `MODERATION` passed with thresholds `0.9`/`0.9`; `BAN` passed with thresholds `0.05`/`0.05`; both responses returned `llm_status="used"`.
- Invalid inputs returned `422`; health checks, logs, response shape, evidence mapping, and risk formula were verified.
- Stack was restored healthy for direct use.

## Web UI Result

Branch: `feat/web-ui`, based on `infra/local-llm` at `1a621a9`.

Changed files:

- `services/coordinator_service/app.py`
- `tests/test_coordinator.py`
- `tests/e2e_validation.py`
- `README.md`
- `docs/worklog/README.md`

New files:

- `web/index.html`
- `web/styles.css`
- `web/app.js`
- `docs/worklog/web-ui-plan.md`

Verified:

- `compileall` passed for `services` and `tests`.
- All service test scripts passed locally.
- Coordinator tests cover `/`, `/static/app.js`, and `/static/styles.css`.
- `GET /` returned `200 text/html`; static JS/CSS returned `200` with correct content types.
- E2E validation with UI checks passed for `MODERATION` at thresholds `0.9`/`0.9`.
- E2E validation with UI checks passed for `BAN` at thresholds `0.05`/`0.05`.
- Both E2E decisions returned `llm_status="used"` with local LLM enabled.
- Stack was restored healthy; UI is available at `http://127.0.0.1:8000/`.

## Final State

`main` is at `608e740`; `origin/main`, `origin/infra/local-llm`, and `origin/feat/web-ui` were verified after push. Docker run instructions for Linux and Windows are in `README.md`, including fresh-PC prerequisites, first-build internet access, `docker compose ps`, port override, and the optional host requirement for E2E validation.

## Current Next Action

None. The implementation is complete, merged into `main`, and pushed.
