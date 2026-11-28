import os
import time

import httpx

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
EXPECTED_DECISION = os.getenv("EXPECTED_DECISION", "MODERATION")
BOT_THRESHOLD = float(os.getenv("BOT_THRESHOLD", "0.5"))
FAKE_NEWS_THRESHOLD = float(os.getenv("FAKE_NEWS_THRESHOLD", "0.5"))
TIMEOUT_SECONDS = float(os.getenv("E2E_TIMEOUT_SECONDS", "60"))

ACCOUNT = {
    "user_id": "account_001",
    "tweets": [
        {"tweet_id": "101", "created_at": "2026-11-20T10:30:00Z"},
        {"tweet_id": "102", "created_at": "2026-11-20T10:35:00Z"},
        {"tweet_id": "103", "created_at": "2026-11-20T11:00:00Z"},
    ],
}


def wait_for_health(client: httpx.Client) -> None:
    deadline = time.monotonic() + TIMEOUT_SECONDS
    last_error = "no response"

    while time.monotonic() < deadline:
        try:
            response = client.get("/health", timeout=2.0)
            if response.status_code == 200 and response.json() == {"status": "ok"}:
                return
            last_error = f"status={response.status_code}, body={response.text}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)

    raise AssertionError(f"Coordinator did not become healthy: {last_error}")


def valid_payload() -> dict:
    return {
        "text": "The election was rigged and officials hid the truth",
        "title": "",
        "account": ACCOUNT,
    }


def assert_probability(name: str, value) -> None:
    assert isinstance(value, (int, float)), f"{name} is not numeric"
    assert 0 <= value <= 1, f"{name} outside [0, 1]: {value}"


def main() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        wait_for_health(client)

        index_response = client.get("/")
        assert index_response.status_code == 200, index_response.text
        assert "Disinformation Detection Console" in index_response.text

        script_response = client.get("/static/app.js")
        assert script_response.status_code == 200, script_response.text
        assert 'fetch("/analyze"' in script_response.text

        response = client.post("/analyze", json=valid_payload())
        assert response.status_code == 200, response.text
        payload = response.json()

        assert set(payload) == {"agent_1", "agent_2", "agent_3"}, payload

        agent_1 = payload["agent_1"]
        agent_2 = payload["agent_2"]
        agent_3 = payload["agent_3"]

        assert set(agent_1) == {"fake_news_probability", "predicted_label"}
        assert set(agent_2) == {
            "user_id",
            "bot_probability",
            "predicted_label",
        }
        assert set(agent_3) == {
            "user_id",
            "final_risk_probability",
            "decision",
            "reason",
            "evidence",
            "llm_status",
        }

        fake_probability = agent_1["fake_news_probability"]
        bot_probability = agent_2["bot_probability"]
        assert_probability("fake_news_probability", fake_probability)
        assert_probability("bot_probability", bot_probability)

        assert agent_1["predicted_label"] == int(
            fake_probability >= FAKE_NEWS_THRESHOLD
        )
        assert agent_2["predicted_label"] == int(bot_probability >= BOT_THRESHOLD)
        assert agent_2["user_id"] == ACCOUNT["user_id"]
        assert agent_3["user_id"] == ACCOUNT["user_id"]

        expected_risk = max(fake_probability, bot_probability)
        assert abs(agent_3["final_risk_probability"] - expected_risk) < 1e-9

        expected_decision = (
            "BAN"
            if bot_probability >= BOT_THRESHOLD
            or fake_probability >= FAKE_NEWS_THRESHOLD
            else "MODERATION"
        )
        assert expected_decision == EXPECTED_DECISION
        assert agent_3["decision"] == EXPECTED_DECISION
        assert agent_3["llm_status"] in {"disabled", "fallback", "used"}
        assert isinstance(agent_3["reason"], str) and agent_3["reason"]
        assert agent_3["evidence"] == {
            "fake_news_probability": fake_probability,
            "bot_probability": bot_probability,
        }

        encoded = str(payload)
        for internal_url in ("agent1:8001", "agent2:8002", "agent3:8003"):
            assert internal_url not in encoded

        invalid_cases = {
            "blank text": {**valid_payload(), "text": "   "},
            "missing text": {"account": ACCOUNT},
            "empty user": {
                **valid_payload(),
                "account": {**ACCOUNT, "user_id": "   "},
            },
            "short account": {
                **valid_payload(),
                "account": {
                    "user_id": "account_002",
                    "tweets": ACCOUNT["tweets"][:2],
                },
            },
            "naive timestamp": {
                **valid_payload(),
                "account": {
                    "user_id": "account_003",
                    "tweets": [
                        {"tweet_id": "301", "created_at": "2026-11-20T10:30:00"},
                        {"tweet_id": "302", "created_at": "2026-11-20T10:35:00"},
                        {"tweet_id": "303", "created_at": "2026-11-20T11:00:00"},
                    ],
                },
            },
        }

        for name, body in invalid_cases.items():
            invalid_response = client.post("/analyze", json=body)
            assert invalid_response.status_code == 422, (
                f"{name} returned {invalid_response.status_code}: "
                f"{invalid_response.text}"
            )

    print(f"End-to-end checks passed for decision {EXPECTED_DECISION}:")
    print(payload)


if __name__ == "__main__":
    main()
