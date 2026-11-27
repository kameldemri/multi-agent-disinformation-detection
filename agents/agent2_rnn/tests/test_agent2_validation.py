
from agents.agent2_rnn.src.predict import predict_accounts

valid_tweets = [
    {"tweet_id": "101", "created_at": "2026-11-20T10:30:00Z"},
    {"tweet_id": "102", "created_at": "2026-11-20T10:35:00Z"},
    {"tweet_id": "103", "created_at": "2026-11-20T11:00:00Z"},
]


def run_test(name, accounts):
    print(f"\n--- {name} ---")

    try:
        result = predict_accounts(accounts)
        print("Unexpected success:", result)
    except (ValueError, TypeError, KeyError) as error:
        print("Correctly rejected:", error)


run_test("Empty account list", [])

run_test(
    "Fewer than 3 tweets",
    [{"user_id": "001", "tweets": valid_tweets[:2]}]
)

run_test(
    "Missing user_id",
    [{"tweets": valid_tweets}]
)

run_test(
    "Missing tweet_id",
    [{
        "user_id": "004",
        "tweets": [
            {"created_at": "2026-11-20T10:30:00Z"},
            *valid_tweets[1:]
        ]
    }]
)

run_test(
    "Missing created_at",
    [{
        "user_id": "005",
        "tweets": [
            {"tweet_id": "101"},
            *valid_tweets[1:]
        ]
    }]
)

run_test(
    "Invalid timestamp",
    [{
        "user_id": "006",
        "tweets": [
            {"tweet_id": "101", "created_at": "invalid-date"},
            *valid_tweets[1:]
        ]
    }]
)

run_test(
    "Timezone-naive timestamp",
    [{
        "user_id": "007",
        "tweets": [
            {"tweet_id": "101", "created_at": "2026-11-20T10:30:00"},
            *valid_tweets[1:]
        ]
    }]
)
