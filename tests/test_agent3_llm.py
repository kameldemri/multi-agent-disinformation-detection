import os

os.environ["LLM_MODE"] = "offline"

from fastapi.testclient import TestClient

from services.agent3_service.app import app
from services.agent3_service.llm import (
    LLMSettings,
    LLMError,
    explain_result,
    parse_reason,
    request_reason,
)
from services.agent3_service.policy import evaluate_decision
from services.agent3_service.schemas import (
    Agent1Evidence,
    Agent2Evidence,
    DecisionRequest,
)


def make_request() -> DecisionRequest:
    return DecisionRequest(
        user_id="account_001",
        agent_1=Agent1Evidence(
            fake_news_probability=0.87,
            predicted_label=1,
        ),
        agent_2=Agent2Evidence(
            user_id="account_001",
            bot_probability=0.93,
            predicted_label=1,
        ),
    )


class FakeResponse:
    def __init__(self, content: str):
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": self.content,
                    }
                }
            ]
        }


class FakeClient:
    def __init__(self, content: str):
        self.content = content
        self.url = None
        self.headers = None
        self.body = None

    def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
        self.url = url
        self.headers = headers
        self.body = json
        return FakeResponse(self.content)


decision = evaluate_decision(make_request())
assert decision.decision == "BAN"
assert decision.final_risk_probability == 0.93
assert decision.llm_status == "disabled"

offline_settings = LLMSettings(
    mode="offline",
    api_key=None,
    base_url="https://example.test/v1",
    model="test-model",
    timeout=1.0,
    max_tokens=50,
)
offline_reason, offline_status = explain_result(decision, offline_settings)
assert offline_reason == decision.reason
assert offline_status == "disabled"

api_settings = LLMSettings(
    mode="api",
    api_key="test-key",
    base_url="https://example.test/v1",
    model="test-model",
    timeout=1.0,
    max_tokens=50,
)
fake_client = FakeClient('```json\n{"reason": "LLM explanation."}\n```')
api_reason, api_status = explain_result(
    decision,
    api_settings,
    client=fake_client,
)
assert api_reason == "LLM explanation."
assert api_status == "used"
assert fake_client.url == "https://example.test/v1/chat/completions"
assert fake_client.headers["Authorization"] == "Bearer test-key"
assert fake_client.body["temperature"] == 0

missing_key_settings = LLMSettings(
    mode="api",
    api_key=None,
    base_url="https://example.test/v1",
    model="test-model",
    timeout=1.0,
    max_tokens=50,
)
fallback_reason, fallback_status = explain_result(
    decision,
    missing_key_settings,
)
assert fallback_reason == decision.reason
assert fallback_status == "fallback"

local_settings = LLMSettings(
    mode="api",
    api_key=None,
    base_url="http://llm:11434/v1",
    model="local-model",
    timeout=1.0,
    max_tokens=50,
    allow_no_key=True,
)
local_client = FakeClient('{"reason": "Local explanation."}')
local_reason, local_status = explain_result(
    decision,
    local_settings,
    client=local_client,
)
assert local_reason == "Local explanation."
assert local_status == "used"
assert local_client.url == "http://llm:11434/v1/chat/completions"
assert "Authorization" not in local_client.headers

plain_client = FakeClient("Reason: Both risks are below their configured thresholds.")
plain_reason, plain_status = explain_result(
    decision,
    local_settings,
    client=plain_client,
)
assert plain_reason == "Both risks are below their configured thresholds."
assert plain_status == "used"

try:
    parse_reason("not json")
except LLMError:
    pass
else:
    raise AssertionError("Malformed LLM output was accepted")

client = TestClient(app)

health_response = client.get("/health")
assert health_response.status_code == 200
assert health_response.json() == {"status": "ok"}

decide_response = client.post("/decide", json=make_request().model_dump(mode="json"))
assert decide_response.status_code == 200

payload = decide_response.json()
assert payload["decision"] == "BAN"
assert payload["final_risk_probability"] == 0.93
assert payload["llm_status"] == "disabled"
assert payload["reason"] == decision.reason

print("Agent 3 LLM explainer checks passed:")
print(payload)
