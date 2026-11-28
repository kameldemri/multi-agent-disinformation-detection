import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from services.coordinator_service.client import load_settings, post_json
from services.coordinator_service.schemas import AnalyzeRequest, AnalyzeResponse

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="Coordinator API", version="1.0.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
SETTINGS = load_settings()


def analyze_request(request: AnalyzeRequest, client=None) -> AnalyzeResponse:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")

    agent_1 = post_json(
        f"{SETTINGS.agent1_url}/predict",
        {"text": text, "title": request.title},
        SETTINGS,
        client=client,
    )

    agent_2 = post_json(
        f"{SETTINGS.agent2_url}/predict",
        request.account.model_dump(mode="json"),
        SETTINGS,
        client=client,
    )

    agent_3 = post_json(
        f"{SETTINGS.agent3_url}/decide",
        {
            "user_id": agent_2["user_id"],
            "agent_1": agent_1,
            "agent_2": agent_2,
        },
        SETTINGS,
        client=client,
    )

    return AnalyzeResponse(
        agent_1=agent_1,
        agent_2=agent_2,
        agent_3=agent_3,
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    return analyze_request(request)


if __name__ == "__main__":
    uvicorn.run(
        "services.coordinator_service.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
