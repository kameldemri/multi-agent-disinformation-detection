from fastapi.testclient import TestClient

from services.agent3_service.app import app
from services.agent3_service.policy import evaluate_decision
from services.agent3_service.schemas import (
    Agent1Evidence,
    Agent2Evidence,
    DecisionRequest,
)


def make_request(
    bot_probability: float,
    fake_news_probability: float,
    user_id: str = "account_001",
    agent_2_user_id: str | None = None,
) -> DecisionRequest:
    return DecisionRequest(
        user_id=user_id,
        agent_1=Agent1Evidence(
            fake_news_probability=fake_news_probability,
            predicted_label=int(fake_news_probability >= 0.5),
        ),
        agent_2=Agent2Evidence(
            user_id=agent_2_user_id or user_id,
            bot_probability=bot_probability,
            predicted_label=int(bot_probability >= 0.5),
        ),
    )


matrix = [
    (0.95, 0.10, "BAN"),
    (0.90, 0.90, "BAN"),
    (0.10, 0.95, "BAN"),
    (0.10, 0.10, "MODERATION"),
    (0.49, 0.49, "MODERATION"),
    (0.50, 0.10, "BAN"),
    (0.10, 0.50, "BAN"),
]

for bot_probability, fake_news_probability, expected_decision in matrix:
    result = evaluate_decision(
        make_request(bot_probability, fake_news_probability)
    )
    assert result.decision == expected_decision
    assert result.final_risk_probability == max(
        bot_probability,
        fake_news_probability,
    )
    assert result.llm_status == "disabled"

client = TestClient(app)

health_response = client.get("/health")
assert health_response.status_code == 200
assert health_response.json() == {"status": "ok"}

request = make_request(0.93, 0.87).model_dump(mode="json")
decide_response = client.post("/decide", json=request)
assert decide_response.status_code == 200

payload = decide_response.json()
assert set(payload) == {
    "user_id",
    "final_risk_probability",
    "decision",
    "reason",
    "evidence",
    "llm_status",
}
assert payload["decision"] == "BAN"
assert payload["final_risk_probability"] == 0.93
assert payload["llm_status"] == "disabled"

mismatch = make_request(0.10, 0.10, agent_2_user_id="account_999").model_dump(
    mode="json"
)
mismatch_response = client.post("/decide", json=mismatch)
assert mismatch_response.status_code == 422

invalid_probability = make_request(0.10, 0.10).model_dump(mode="json")
invalid_probability["agent_1"]["fake_news_probability"] = 1.5
invalid_response = client.post("/decide", json=invalid_probability)
assert invalid_response.status_code == 422

print("Agent 3 policy checks passed:")
print(payload)
