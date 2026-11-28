from agents.agent2_rnn.src.predict import predict_accounts


accounts = [
    {
        "user_id": "account_001",
        "tweets": [
            {
                "tweet_id": "101",
                "created_at": "2026-11-20T10:30:00Z",
            },
            {
                "tweet_id": "102",
                "created_at": "2026-11-20T10:35:00Z",
            },
            {
                "tweet_id": "103",
                "created_at": "2026-11-20T11:00:00Z",
            },
        ],
    },
    {
        "user_id": "account_002",
        "tweets": [
            {
                "tweet_id": "201",
                "created_at": "2026-11-20T08:00:00+01:00",
            },
            {
                "tweet_id": "202",
                "created_at": "2026-11-20T08:10:00+01:00",
            },
            {
                "tweet_id": "203",
                "created_at": "2026-11-20T09:00:00+01:00",
            },
        ],
    },
]

predictions = predict_accounts(accounts)

print("Predictions:")

for prediction in predictions:
    print(prediction)

print("\nNumber of input accounts:", len(accounts))
print("Number of predictions:", len(predictions))

input_ids = [account["user_id"] for account in accounts]
output_ids = [prediction["user_id"] for prediction in predictions]

print("User IDs preserved:", input_ids == output_ids)