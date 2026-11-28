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
- Current branch: `feat/agent2-service`.
- Planning docs are committed locally but not pushed.
- Documentation-only work is folded into the planning branch; no throwaway phase branch.
- Implementation phases get professional branches such as `feat/agent1-service`.
- Do not merge into `main` until the user confirms.

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
- Docker structure: one service per agent plus one coordinator service.
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

## Current Next Action

User confirms `feat/agent2-service`; then create `feat/agent3-policy` and implement the deterministic Agent 3 core.
