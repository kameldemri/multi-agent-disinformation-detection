from fastapi import HTTPException
from fastapi.testclient import TestClient

from services.coordinator_service.app import SETTINGS, analyze_request, app
from services.coordinator_service.schemas import AnalyzeRequest


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self.payload = payload

    def json(self) -> dict:
        return self.payload


class FakeClient:
    def __init__(self, responses: dict):
        self.responses = responses
        self.calls = []

    def post(self, url: str, json: dict) -> FakeResponse:
        self.calls.append((url, json))
        return self.responses[url]


agent_1_payload = {
    "fake_news_probability": 0.45518806499284004,
    "predicted_label": 0,
}
agent_2_payload = {
    "user_id": "account_001",
    "bot_probability": 0.1048131138086319,
    "predicted_label": 0,
}
agent_3_payload = {
    "user_id": "account_001",
    "final_risk_probability": 0.45518806499284004,
    "decision": "MODERATION",
    "reason": "Both content and behavioral risks are below their configured thresholds.",
    "evidence": {
        "fake_news_probability": 0.45518806499284004,
        "bot_probability": 0.1048131138086319,
    },
    "llm_status": "disabled",
}

responses = {
    f"{SETTINGS.agent1_url}/predict": FakeResponse(200, agent_1_payload),
    f"{SETTINGS.agent2_url}/predict": FakeResponse(200, agent_2_payload),
    f"{SETTINGS.agent3_url}/decide": FakeResponse(200, agent_3_payload),
}
fake_client = FakeClient(responses)

request = AnalyzeRequest(
    text="The election was rigged and officials hid the truth",
    title="",
    account={
        "user_id": "account_001",
        "tweets": [
            {"tweet_id": "101", "created_at": "2026-11-20T10:30:00Z"},
            {"tweet_id": "102", "created_at": "2026-11-20T10:35:00Z"},
            {"tweet_id": "103", "created_at": "2026-11-20T11:00:00Z"},
        ],
    },
)

result = analyze_request(request, client=fake_client)
assert result.agent_1.model_dump() == agent_1_payload
assert result.agent_2.model_dump() == agent_2_payload
assert result.agent_3.model_dump() == agent_3_payload
assert [url for url, _ in fake_client.calls] == [
    f"{SETTINGS.agent1_url}/predict",
    f"{SETTINGS.agent2_url}/predict",
    f"{SETTINGS.agent3_url}/decide",
]
assert fake_client.calls[2][1]["user_id"] == "account_001"

error_responses = {
    f"{SETTINGS.agent1_url}/predict": FakeResponse(200, agent_1_payload),
    f"{SETTINGS.agent2_url}/predict": FakeResponse(
        422,
        {"detail": "Account 001 must contain at least 3 tweets."},
    ),
}
error_client = FakeClient(error_responses)

try:
    analyze_request(request, client=error_client)
except HTTPException as exc:
    assert exc.status_code == 422
else:
    raise AssertionError("Agent 2 422 was not propagated")

client = TestClient(app)

health_response = client.get("/health")
assert health_response.status_code == 200
assert health_response.json() == {"status": "ok"}

invalid_response = client.post(
    "/analyze",
    json={
        "text": "   ",
        "account": {
            "user_id": "account_001",
            "tweets": [
                {"tweet_id": "101", "created_at": "2026-11-20T10:30:00Z"},
                {"tweet_id": "102", "created_at": "2026-11-20T10:35:00Z"},
                {"tweet_id": "103", "created_at": "2026-11-20T11:00:00Z"},
            ],
        },
    },
)
assert invalid_response.status_code == 422

print("Coordinator API checks passed:")
print(result.model_dump())
