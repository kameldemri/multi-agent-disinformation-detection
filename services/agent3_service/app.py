import os

import uvicorn
from fastapi import FastAPI, HTTPException

from services.agent3_service.llm import explain_result, load_settings
from services.agent3_service.policy import evaluate_decision
from services.agent3_service.schemas import DecisionRequest, DecisionResponse

app = FastAPI(title="Agent 3 Decision Service", version="1.0.0")
LLM_SETTINGS = load_settings()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/decide", response_model=DecisionResponse)
def decide(request: DecisionRequest) -> DecisionResponse:
    user_id = request.user_id.strip()
    agent_2_user_id = request.agent_2.user_id.strip()

    if not user_id or not agent_2_user_id:
        raise HTTPException(status_code=422, detail="user_id must not be empty")

    if user_id != agent_2_user_id:
        raise HTTPException(status_code=422, detail="user_id values must match")

    request.user_id = user_id
    request.agent_2.user_id = agent_2_user_id

    decision = evaluate_decision(request)
    reason, llm_status = explain_result(decision, LLM_SETTINGS)
    decision.reason = reason
    decision.llm_status = llm_status
    return decision


if __name__ == "__main__":
    uvicorn.run(
        "services.agent3_service.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8003")),
    )
