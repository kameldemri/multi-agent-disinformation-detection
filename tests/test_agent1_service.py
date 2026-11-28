from fastapi.testclient import TestClient

from services.agent1_service.app import FAKE_NEWS_THRESHOLD, app

client = TestClient(app)

health_response = client.get("/health")
assert health_response.status_code == 200
assert health_response.json() == {"status": "ok"}

text = "The election was rigged and officials hid the truth"
predict_response = client.post("/predict", json={"text": text})
assert predict_response.status_code == 200

payload = predict_response.json()
assert set(payload) == {"fake_news_probability", "predicted_label"}
assert 0 <= payload["fake_news_probability"] <= 1
assert payload["predicted_label"] in (0, 1)
assert payload["predicted_label"] == int(
    payload["fake_news_probability"] >= FAKE_NEWS_THRESHOLD
)

empty_response = client.post("/predict", json={"text": "   "})
assert empty_response.status_code == 422

missing_response = client.post("/predict", json={})
assert missing_response.status_code == 422

print("Agent 1 service checks passed:")
print(payload)
