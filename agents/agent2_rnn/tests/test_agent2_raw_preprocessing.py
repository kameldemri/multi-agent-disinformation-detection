from agents.agent2_rnn.src.predict import (
    parse_tweet_timestamp,
    extract_temporal_features
)

from agents.agent2_rnn.src.predict import (
    parse_tweet_timestamp,
    extract_temporal_features,
    transform_temporal_features
)

import numpy as np

from agents.agent2_rnn.src.predict import (
    parse_tweet_timestamp,
    extract_temporal_features,
    transform_temporal_features,
    scale_temporal_features
)


from agents.agent2_rnn.src.predict import (
    parse_tweet_timestamp,
    extract_temporal_features,
    transform_temporal_features,
    scale_temporal_features,
    preprocess_account_tweets
)

# Test 1: Different timestamp formats and timezone offsets.
account = {
    "user_id": "123456789",
    "tweets": [
        {
            "tweet_id": "103",
            "created_at": "2018-10-10T22:00:00+01:00"
        },
        {
            "tweet_id": "101",
            "created_at": "Wed Oct 10 20:19:24 +0000 2018"
        },
        {
            "tweet_id": "102",
            "created_at": "2018-10-10T21:00:00Z"
        }
    ]
}

features = extract_temporal_features(account)

# Test 4: Transform raw temporal features into five model features.
transformed = transform_temporal_features(features)

# Test 5: Apply the training scaler.
scaled = scale_temporal_features(transformed)

print("\nScaled feature matrix:")
print(scaled)

print("\nScaled matrix shape:", scaled.shape)
print("Scaled data type:", scaled.dtype)
print("All scaled values finite:", np.isfinite(scaled).all())

print("\nTransformed feature matrix:")
print(transformed)

print("\nMatrix shape:", transformed.shape)
print("Data type:", transformed.dtype)
print("All values finite:", np.isfinite(transformed).all())

for feature in features:
    print(feature)

# Test 2: Reject fewer than 3 tweets.
try:
    extract_temporal_features({
        "user_id": "123",
        "tweets": account["tweets"][:2]
    })
except ValueError as error:
    print("\nMinimum-length validation:", error)

# Test 3: Reject a timezone-naive timestamp.
try:
    parse_tweet_timestamp("2018-10-10T20:19:24")
except ValueError as error:
    print("\nTimezone validation:", error)

# Test 6: Complete raw-account preprocessing pipeline.
sequence = preprocess_account_tweets(account)

print("\nComplete preprocessed sequence:")
print(sequence)

print("\nSequence shape:", sequence.shape)
print("Sequence dtype:", sequence.dtype)
print("All values finite:", np.isfinite(sequence).all())