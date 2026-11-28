import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.agent2_rnn.src.predict import load_model, predict_accounts


def _load_threshold() -> float:
    raw_value = os.getenv("BOT_THRESHOLD", "0.5")
    try:
        threshold = float(raw_value)
    except ValueError as exc:
        raise RuntimeError("BOT_THRESHOLD must be a float") from exc
    if not 0 <= threshold <= 1:
        raise RuntimeError("BOT_THRESHOLD must be between 0 and 1")
    return threshold


BOT_THRESHOLD = _load_threshold()
_MODEL = None
_DEVICE = None

app = FastAPI(title="Agent 2 Bot Detection Service", version="1.0.0")


class TweetRequest(BaseModel):
    tweet_id: str | int
    created_at: str = Field(min_length=1)


class AccountRequest(BaseModel):
    user_id: str | int
    tweets: list[TweetRequest] = Field(min_length=3)


class PredictResponse(BaseModel):
    user_id: str
    bot_probability: float = Field(ge=0, le=1)
    predicted_label: int


def _get_model():
    global _MODEL, _DEVICE
    if _MODEL is None:
        _MODEL, _DEVICE, _ = load_model()
    return _MODEL, _DEVICE


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(account: AccountRequest) -> dict:
    model, device = _get_model()
    raw_account = {
        "user_id": str(account.user_id).strip(),
        "tweets": [
            {
                "tweet_id": str(tweet.tweet_id),
                "created_at": tweet.created_at,
            }
            for tweet in account.tweets
        ],
    }

    if not raw_account["user_id"]:
        raise HTTPException(status_code=422, detail="user_id must not be empty")

    try:
        predictions = predict_accounts(
            [raw_account],
            model=model,
            device=device,
            threshold=BOT_THRESHOLD,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return predictions[0]


if __name__ == "__main__":
    uvicorn.run(
        "services.agent2_service.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8002")),
    )
