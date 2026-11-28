from typing import Literal

from pydantic import BaseModel, Field


class TweetInput(BaseModel):
    tweet_id: str | int
    created_at: str = Field(min_length=1)


class AccountInput(BaseModel):
    user_id: str | int
    tweets: list[TweetInput] = Field(min_length=3)


class Agent1Output(BaseModel):
    fake_news_probability: float = Field(ge=0, le=1)
    predicted_label: int = Field(ge=0, le=1)


class Agent2Output(BaseModel):
    user_id: str
    bot_probability: float = Field(ge=0, le=1)
    predicted_label: int = Field(ge=0, le=1)


class Agent3Evidence(BaseModel):
    fake_news_probability: float
    bot_probability: float


class Agent3Output(BaseModel):
    user_id: str
    final_risk_probability: float = Field(ge=0, le=1)
    decision: Literal["BAN", "MODERATION"]
    reason: str
    evidence: Agent3Evidence
    llm_status: Literal["used", "fallback", "disabled"]


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)
    title: str = ""
    account: AccountInput


class AnalyzeResponse(BaseModel):
    agent_1: Agent1Output
    agent_2: Agent2Output
    agent_3: Agent3Output
