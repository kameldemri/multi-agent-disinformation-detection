
import numpy as np
import pandas as pd
from pathlib import Path

data_dir = Path("data/processed/agent2")

# Read account split information
splits = pd.read_csv(
    data_dir / "account_splits.csv",
    dtype={"user_id": "string"}
)

# Count tweets per account without loading the entire CSV into memory
tweet_counts = {}

for chunk in pd.read_csv(
    data_dir / "temporal_features.csv",
    usecols=["user_id"],
    dtype={"user_id": "string"},
    chunksize=100_000
):
    counts = chunk.groupby("user_id").size()

    for user_id, count in counts.items():
        tweet_counts[user_id] = (
            tweet_counts.get(user_id, 0) + count
        )

# Compare expected account sequence lengths with NPZ lengths
for split in ["train", "validation", "test"]:

    expected = (
        splits[splits["split"] == split]
        .sort_values("user_id")
        .reset_index(drop=True)
    )

    expected_lengths = (
        expected["user_id"]
        .map(tweet_counts)
        .to_numpy(dtype=np.int64)
    )

    data = np.load(
        data_dir / f"{split}_scaled_sequences.npz",
        allow_pickle=True
    )

    actual_lengths = np.array(
        [len(sequence) for sequence in data["sequences"]]
    )

    print(f"\n{split.upper()}")
    print("Expected sequence lengths:", len(expected_lengths))
    print("Stored sequence lengths:", len(actual_lengths))
    print(
        "Sequence lengths match:",
        np.array_equal(expected_lengths, actual_lengths)
    )

    data.close()
