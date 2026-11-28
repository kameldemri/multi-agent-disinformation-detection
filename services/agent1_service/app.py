import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.agent1_naive_bayes.src.predict import score_text


def _load_threshold() -> float:
    raw_value = os.getenv("FAKE_NEWS_THRESHOLD", "0.5")
    try:
        threshold = float(raw_value)
    except ValueError as exc:
        raise RuntimeError("FAKE_NEWS_THRESHOLD must be a float") from exc
    if not 0 <= threshold <= 1:
        raise RuntimeError("FAKE_NEWS_THRESHOLD must be between 0 and 1")
    return threshold


FAKE_NEWS_THRESHOLD = _load_threshold()

app = FastAPI(title="Agent 1 Fake News Service", version="1.0.0")


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)
    title: str = ""


class PredictResponse(BaseModel):
    fake_news_probability: float = Field(ge=0, le=1)
    predicted_label: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")

    fake_news_probability = score_text(text, request.title)
    return {
        "fake_news_probability": fake_news_probability,
        "predicted_label": int(fake_news_probability >= FAKE_NEWS_THRESHOLD),
    }


if __name__ == "__main__":
    uvicorn.run(
        "services.agent1_service.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8001")),
    )
