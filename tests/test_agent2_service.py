from fastapi.testclient import TestClient

from services.agent2_service.app import BOT_THRESHOLD, app

client = TestClient(app)

health_response = client.get("/health")
assert health_response.status_code == 200
assert health_response.json() == {"status": "ok"}

account = {
    "user_id": "account_001",
    "tweets": [
        {"tweet_id": "101", "created_at": "2026-11-20T10:30:00Z"},
        {"tweet_id": "102", "created_at": "2026-11-20T10:35:00Z"},
        {"tweet_id": "103", "created_at": "2026-11-20T11:00:00Z"},
    ],
}

predict_response = client.post("/predict", json=account)
assert predict_response.status_code == 200

payload = predict_response.json()
assert set(payload) == {"user_id", "bot_probability", "predicted_label"}
assert payload["user_id"] == "account_001"
assert 0 <= payload["bot_probability"] <= 1
assert payload["predicted_label"] in (0, 1)
assert payload["predicted_label"] == int(
    payload["bot_probability"] >= BOT_THRESHOLD
)

short_account = {
    "user_id": "account_002",
    "tweets": [
        {"tweet_id": "201", "created_at": "2026-11-20T10:30:00Z"},
        {"tweet_id": "202", "created_at": "2026-11-20T10:35:00Z"},
    ],
}
short_response = client.post("/predict", json=short_account)
assert short_response.status_code == 422

naive_account = {
    "user_id": "account_003",
    "tweets": [
        {"tweet_id": "301", "created_at": "2026-11-20T10:30:00"},
        {"tweet_id": "302", "created_at": "2026-11-20T10:35:00"},
        {"tweet_id": "303", "created_at": "2026-11-20T11:00:00"},
    ],
}
naive_response = client.post("/predict", json=naive_account)
assert naive_response.status_code == 422

empty_user_response = client.post(
    "/predict",
    json={**account, "user_id": "   "},
)
assert empty_user_response.status_code == 422

print("Agent 2 service checks passed:")
print(payload)
