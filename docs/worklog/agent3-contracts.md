# Agent 3 Contracts

Folded into planning branch: `plan/agent3-decision-layer`

Status: merged into `main` at `3ccc516`; pushed.

## Goal

Freeze the exact input/output contracts between Agent 1, Agent 2, Agent 3, and the coordinator.

## Result

Contracts are frozen below. No application code changed.

## Agent 1 Raw API

File: `agents/agent1_naive_bayes/src/predict.py`

```python
score_text(text: str, title: str = "") -> float
score_texts(texts: list[str], titles: list[str] | None = None) -> list[float]
explain_text(text: str, title: str = "", n: int = 8) -> dict
```

Raw output type: `float`.

Meaning:

- `0.0` = surely true.
- `1.0` = surely fake.
- `0.5` = undecided / no known words.
- CLI labels `FAKE` if `p >= 0.5`, else `TRUE`.

Important limitation:

- Trained on English news articles, not tweets.

## Agent 1 Service Contract

Endpoint planned: `POST /predict`

Input:

```json
{
  "text": "The election was rigged and officials hid the truth",
  "title": ""
}
```

Output:

```json
{
  "fake_news_probability": 0.45518806499284004,
  "predicted_label": 0
}
```

Rules:

- `text` must be non-empty.
- `title` is optional and defaults to `""`.
- `fake_news_probability` is the float returned by `score_text(text, title)`.
- `predicted_label = int(fake_news_probability >= FAKE_NEWS_THRESHOLD)`.
- Default `FAKE_NEWS_THRESHOLD=0.5`.
- Empty `text` returns `422`.

## Agent 2 Raw API

File: `agents/agent2_rnn/src/predict.py`

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

Raw input is a list of accounts. Raw output is a list of prediction dictionaries.

## Agent 2 Service Contract

Endpoint planned: `POST /predict`

Input is one account object, not a list:

```json
{
  "user_id": "account_001",
  "tweets": [
    {"tweet_id": "101", "created_at": "2026-11-20T10:30:00Z"},
    {"tweet_id": "102", "created_at": "2026-11-20T10:35:00Z"},
    {"tweet_id": "103", "created_at": "2026-11-20T11:00:00Z"}
  ]
}
```

Output:

```json
{
  "user_id": "account_001",
  "bot_probability": 0.1048131138086319,
  "predicted_label": 0
}
```

Rules:

- `user_id` must be non-empty.
- At least 3 tweets are required.
- Each tweet needs `tweet_id` and `created_at`.
- `created_at` must include timezone information.
- Supported timestamp formats include Twitter format and ISO 8601 with timezone.
- Tweet text is not used by Agent 2.
- Invalid input returns `422`.
- Service adapter calls `predict_accounts([account])` and returns element `0`.

## Agent 3 Input Contract

Agent 3 is per account.

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

Validation rules:

- `user_id` must be present.
- `agent_1.fake_news_probability` must be a float in `[0, 1]`.
- `agent_2.bot_probability` must be a float in `[0, 1]`.
- `agent_2.user_id` must match top-level `user_id`.
- Missing or invalid evidence returns `422`.
- Agent 3 decision uses probabilities, not `predicted_label`.

## Agent 3 Decision Contract

Thresholds:

- `BOT_THRESHOLD`, default `0.5`.
- `FAKE_NEWS_THRESHOLD`, default `0.5`.

Policy:

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

Allowed decisions:

- `BAN`
- `MODERATION`

No other enforcement actions are introduced.

## Agent 3 Output Contract

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

LLM restrictions:

- LLM may only generate `reason`.
- LLM may not change `decision`.
- LLM may not change `final_risk_probability`.
- LLM may not invent or modify evidence.
- Malformed LLM output falls back to deterministic `reason`.

## Coordinator Public Contract

Endpoint planned: `POST /analyze`

Input:

```json
{
  "text": "The election was rigged and officials hid the truth",
  "title": "",
  "account": {
    "user_id": "account_001",
    "tweets": [
      {"tweet_id": "101", "created_at": "2026-11-20T10:30:00Z"},
      {"tweet_id": "102", "created_at": "2026-11-20T10:35:00Z"},
      {"tweet_id": "103", "created_at": "2026-11-20T11:00:00Z"}
    ]
  }
}
```

Output:

```json
{
  "agent_1": {
    "fake_news_probability": 0.45518806499284004,
    "predicted_label": 0
  },
  "agent_2": {
    "user_id": "account_001",
    "bot_probability": 0.1048131138086319,
    "predicted_label": 0
  },
  "agent_3": {
    "user_id": "account_001",
    "final_risk_probability": 0.1048131138086319,
    "decision": "MODERATION",
    "reason": "Both content and behavioral risks are below their configured thresholds.",
    "evidence": {
      "fake_news_probability": 0.45518806499284004,
      "bot_probability": 0.1048131138086319
    },
    "llm_status": "disabled"
  }
}
```

## Commands Run

```bash
PYTHONPATH=. .venv/bin/python -c "from agents.agent1_naive_bayes.src.predict import score_text; print(score_text('The election was rigged and officials hid the truth'))"
PYTHONPATH=. .venv/bin/python -c "from agents.agent2_rnn.src.predict import predict_accounts; print(predict_accounts([{'user_id':'account_001','tweets':[{'tweet_id':'101','created_at':'2026-11-20T10:30:00Z'},{'tweet_id':'102','created_at':'2026-11-20T10:35:00Z'},{'tweet_id':'103','created_at':'2026-11-20T11:00:00Z'}]}]))"
```

## Verification Results

- Agent 1 raw score: `0.45518806499284004`.
- Agent 2 raw output: `[{'user_id': 'account_001', 'bot_probability': 0.1048131138086319, 'predicted_label': 0}]`.
- No unresolved field names remain for phases 02-06.

## Next Step

User confirms the plan; then create `feat/agent1-service`.
