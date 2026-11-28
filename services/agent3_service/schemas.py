from typing import Literal

from pydantic import BaseModel, Field


class Agent1Evidence(BaseModel):
    fake_news_probability: float = Field(ge=0, le=1)
    predicted_label: int = Field(ge=0, le=1)


class Agent2Evidence(BaseModel):
    user_id: str = Field(min_length=1)
    bot_probability: float = Field(ge=0, le=1)
    predicted_label: int = Field(ge=0, le=1)


class DecisionRequest(BaseModel):
    user_id: str = Field(min_length=1)
    agent_1: Agent1Evidence
    agent_2: Agent2Evidence


class Agent3Evidence(BaseModel):
    fake_news_probability: float
    bot_probability: float


class DecisionResponse(BaseModel):
    user_id: str
    final_risk_probability: float = Field(ge=0, le=1)
    decision: Literal["BAN", "MODERATION"]
    reason: str
    evidence: Agent3Evidence
    llm_status: Literal["used", "fallback", "disabled"]
