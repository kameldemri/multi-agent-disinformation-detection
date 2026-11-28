
import pandas as pd
from pathlib import Path

import torch

from agents.agent2_rnn.src.model import AccountBiGRU

import numpy as np

from torch.nn.utils.rnn import pad_sequence

from datetime import datetime, timezone


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "agents"
    / "agent2_rnn"
    / "artifacts"
    / "best_bigru_safe.pt"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "agent2"
    / "temporal_scaler.npz"
)

ACCOUNT_SPLITS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "agent2"
    / "account_splits.csv"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "agent2"
)

def parse_tweet_timestamp(value):
    """
    Parse a tweet creation timestamp and convert it to UTC.

    Supported formats:
    - Twitter original format:
      Wed Oct 10 20:19:24 +0000 2018

    - ISO 8601:
      2018-10-10T20:19:24Z
      2018-10-10T21:19:24+01:00

    Timezone-naive timestamps are rejected.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Tweet timestamp must be a non-empty string.")

    value = value.strip()

    # Try the original Twitter timestamp format.
    try:
        timestamp = datetime.strptime(
            value,
            "%a %b %d %H:%M:%S %z %Y"
        )
    except ValueError:
        # If it is not a Twitter timestamp, try ISO 8601.
        try:
            timestamp = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                f"Unsupported timestamp format: {value}"
            ) from exc

    # Reject timestamps without timezone information.
    if timestamp.tzinfo is None:
        raise ValueError(
            f"Timezone information is required: {value}"
        )

    return timestamp.astimezone(timezone.utc)

def extract_temporal_features(account):
    """
    Extract raw temporal features from one account's raw tweets.

    Expected input:
    {
        "user_id": "...",
        "tweets": [
            {
                "tweet_id": "...",
                "created_at": "..."
            }
        ]
    }

    Returns a list of dictionaries containing:
    tweet_id, gap_seconds, hour, and day_of_week.
    """

    if not isinstance(account, dict):
        raise ValueError("Account must be a dictionary.")

    user_id = account.get("user_id")
    tweets = account.get("tweets")

    if user_id is None or str(user_id).strip() == "":
        raise ValueError("Account must have a valid user_id.")

    if not isinstance(tweets, list):
        raise ValueError("Account tweets must be provided as a list.")

    if len(tweets) < 3:
        raise ValueError(
            f"Account {user_id} must contain at least 3 tweets."
        )

    parsed_tweets = []
    seen_tweet_ids = set()

    for index, tweet in enumerate(tweets):

        if not isinstance(tweet, dict):
            raise ValueError(
                f"Tweet at position {index} must be a dictionary."
            )

        tweet_id = tweet.get("tweet_id")
        created_at = tweet.get("created_at")

        if tweet_id is None or str(tweet_id).strip() == "":
            raise ValueError(
                f"Tweet at position {index} has no valid tweet_id."
            )

        tweet_id_key = str(tweet_id)
        if tweet_id_key in seen_tweet_ids:
            raise ValueError(
                f"Duplicate tweet_id found in account {user_id}: "
                f"{tweet_id_key}"
            )

        seen_tweet_ids.add(tweet_id_key)

        if created_at is None:
            raise ValueError(
                f"Tweet {tweet_id} has no created_at timestamp."
            )

        timestamp_utc = parse_tweet_timestamp(created_at)

        parsed_tweets.append({
            "tweet_id": tweet_id,
            "timestamp_utc": timestamp_utc
        })

    # Match the chronological ordering used during training:
    # timestamp first, tweet_id as the tie-breaker.
    def tweet_sort_key(tweet):
        tweet_id = tweet["tweet_id"]

        # Twitter IDs are normally numeric strings.
        # Convert them numerically for a deterministic ascending order.
        try:
            tweet_id_key = (0, int(str(tweet_id)))
        except (TypeError, ValueError):
            # Fallback for non-numeric IDs.
            tweet_id_key = (1, str(tweet_id))

        return tweet["timestamp_utc"], tweet_id_key

    parsed_tweets.sort(key=tweet_sort_key)

    temporal_features = []
    previous_timestamp = None

    for tweet in parsed_tweets:

        timestamp = tweet["timestamp_utc"]

        if previous_timestamp is None:
            gap_seconds = 0.0
        else:
            gap_seconds = (
                timestamp - previous_timestamp
            ).total_seconds()

        temporal_features.append({
            "tweet_id": str(tweet["tweet_id"]),
            "gap_seconds": gap_seconds,
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday()
        })

        previous_timestamp = timestamp

    return temporal_features

def transform_temporal_features(temporal_features):
    """
    Transform raw temporal values into the five features
    expected by the trained Agent 2 BiGRU.

    Feature order:
    1. log1p(gap_seconds)
    2. hour_sin
    3. hour_cos
    4. day_sin
    5. day_cos

    Returns:
        numpy array with shape [number_of_tweets, 5].
    """

    if not temporal_features:
        raise ValueError("Temporal features cannot be empty.")

    transformed = []

    for feature in temporal_features:

        gap_seconds = feature["gap_seconds"]
        hour = feature["hour"]
        day_of_week = feature["day_of_week"]

        if gap_seconds < 0:
            raise ValueError("Time gaps cannot be negative.")

        if not 0 <= hour <= 23:
            raise ValueError("Hour must be between 0 and 23.")

        if not 0 <= day_of_week <= 6:
            raise ValueError("Day of week must be between 0 and 6.")

        log_gap = np.log1p(gap_seconds)

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)

        transformed.append([
            log_gap,
            hour_sin,
            hour_cos,
            day_sin,
            day_cos
        ])

    return np.asarray(transformed, dtype=np.float32)

def scale_temporal_features(features, scaler_path=SCALER_PATH):
    """
    Standardize the five temporal features using the scaler
    fitted during Agent 2 training.

    Scaling is performed as:

        scaled = (features - train_mean) / train_std

    Returns:
        numpy array with shape [number_of_tweets, 5].
    """

    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2 or features.shape[1] != 5:
        raise ValueError(
            f"Expected feature matrix with shape [T, 5], "
            f"got {features.shape}."
        )

    if not np.isfinite(features).all():
        raise ValueError("Features contain non-finite values.")

    scaler = np.load(scaler_path)

    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)

    if mean.shape != (5,):
        raise ValueError(
            f"Expected scaler mean shape (5,), got {mean.shape}."
        )

    if std.shape != (5,):
        raise ValueError(
            f"Expected scaler std shape (5,), got {std.shape}."
        )

    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Scaler contains non-finite values.")

    if np.any(std <= 0):
        raise ValueError("Scaler standard deviations must be positive.")

    scaled = (features - mean) / std

    if not np.isfinite(scaled).all():
        raise ValueError("Scaled features contain non-finite values.")

    return scaled.astype(np.float32)

def preprocess_account_tweets(account, scaler_path=SCALER_PATH):
    """
    Convert one account's raw tweets into the scaled temporal
    sequence expected by the Agent 2 BiGRU.

    Input:
        {
            "user_id": "...",
            "tweets": [
                {
                    "tweet_id": "...",
                    "created_at": "..."
                },
                ...
            ]
        }

    Returns:
        numpy array with shape [T, 5], dtype float32.
    """

    temporal_features = extract_temporal_features(account)

    transformed_features = transform_temporal_features(
        temporal_features
    )

    scaled_features = scale_temporal_features(
        transformed_features,
        scaler_path=scaler_path
    )

    return scaled_features

# ============================================================
# MODEL LOADING
# ============================================================

def load_model(checkpoint_path=CHECKPOINT_PATH, device=None):
    """
    Load the trained Agent 2 BiGRU model.

    Returns:
        model: Loaded AccountBiGRU in evaluation mode.
        device: Device used for inference.
        checkpoint: Loaded checkpoint dictionary.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # Load checkpoint using safe weights-only loading.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    # Recover the original model configuration.
    config = checkpoint["config"]

    model = AccountBiGRU(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    ).to(device)

    # Restore trained parameters.
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Disable training behavior during inference.
    model.eval()

    return model, device, checkpoint


def load_scaler(scaler_path=SCALER_PATH):
    """
    Load the temporal scaler fitted on training data.

    Returns:
        mean: Training feature means.
        std: Training feature standard deviations.
    """

    scaler_path = Path(scaler_path)

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}"
        )

    with np.load(scaler_path) as data:
        if "mean" not in data or "std" not in data:
            raise ValueError(
                "Scaler must contain 'mean' and 'std'."
            )

        mean = data["mean"].astype(np.float32)
        std = data["std"].astype(np.float32)

    if mean.shape != (5,) or std.shape != (5,):
        raise ValueError(
            "Expected scaler mean and std with shape (5,)."
        )

    if not np.all(np.isfinite(mean)):
        raise ValueError("Scaler mean contains invalid values.")

    if not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("Scaler std contains invalid values.")

    return mean, std


def load_split_data(split="test"):
    """
    Load scaled account sequences and associate each sequence
    with its original user_id.

    The NPZ files do not contain user IDs. Their account order
    is reconstructed by sorting account_splits.csv by user_id.

    Returns:
        records: List of dictionaries containing:
            - user_id
            - sequence
            - label
            - original_index
    """

    # --------------------------------------------------
    # 1. Validate requested split
    # --------------------------------------------------

    valid_splits = {"train", "validation", "test"}

    if split not in valid_splits:
        raise ValueError(
            f"Invalid split '{split}'. "
            f"Choose from {valid_splits}."
        )

    npz_path = (
        PROCESSED_DATA_DIR
        / f"{split}_scaled_sequences.npz"
    )

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Sequence file not found: {npz_path}"
        )

    if not ACCOUNT_SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"Account splits file not found: "
            f"{ACCOUNT_SPLITS_PATH}"
        )

    # --------------------------------------------------
    # 2. Recover account IDs and labels
    # --------------------------------------------------

    account_splits = pd.read_csv(
        ACCOUNT_SPLITS_PATH,
        dtype={"user_id": "string"}
    )

    expected_accounts = (
        account_splits[
            account_splits["split"] == split
        ]
        .sort_values("user_id")
        .reset_index(drop=True)
    )

    if expected_accounts["user_id"].isna().any():
        raise ValueError(
            "Missing user IDs in account_splits.csv."
        )

    if expected_accounts["user_id"].duplicated().any():
        raise ValueError(
            "Duplicate user IDs found in account_splits.csv."
        )

    # --------------------------------------------------
    # 3. Load already-scaled sequences and labels
    # --------------------------------------------------

    with np.load(npz_path, allow_pickle=True) as data:

        sequences = data["sequences"]
        labels = data["labels"]

    if len(sequences) != len(expected_accounts):
        raise ValueError(
            "Sequence count does not match account count."
        )

    if len(labels) != len(expected_accounts):
        raise ValueError(
            "Label count does not match account count."
        )

    expected_labels = (
        expected_accounts["label"].to_numpy()
    )

    if not np.array_equal(expected_labels, labels):
        raise ValueError(
            "NPZ labels do not match account_splits.csv. "
            "Cannot safely associate user IDs."
        )

    # --------------------------------------------------
    # 4. Build account-to-sequence mapping
    # --------------------------------------------------

    records = []

    for index, (user_id, sequence, label) in enumerate(
        zip(
            expected_accounts["user_id"],
            sequences,
            labels
        )
    ):

        sequence = np.asarray(
            sequence,
            dtype=np.float32
        )

        if sequence.ndim != 2 or sequence.shape[1] != 5:
            raise ValueError(
                f"Invalid sequence shape for account {user_id}: "
                f"{sequence.shape}"
            )

        if len(sequence) == 0:
            raise ValueError(
                f"Empty sequence for account {user_id}."
            )

        if not np.all(np.isfinite(sequence)):
            raise ValueError(
                f"Non-finite features found for account {user_id}."
            )

        records.append({
            "user_id": str(user_id),
            "sequence": sequence,
            "label": int(label),
            "original_index": index
        })

    return records

def predict_records(records, model, device, batch_size=16, threshold=0.5):
    """
    Generate bot probabilities and predicted labels for account records.

    Sequences are sorted by descending length before batching,
    then predictions are restored to the original account order.

    Returns:
        List of dictionaries containing:
            - user_id
            - bot_probability
            - predicted_label
    """
    if not records:
        raise ValueError("The records list is empty.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    # Keep each record's original position.
    indexed_records = list(enumerate(records))

    # The BiGRU requires descending sequence lengths.
    indexed_records.sort(
        key=lambda item: len(item[1]["sequence"]),
        reverse=True
    )

    predictions = []

    model.eval()

    with torch.inference_mode():
        for start in range(0, len(indexed_records), batch_size):

            batch = indexed_records[start:start + batch_size]

            # Convert sequences to tensors.
            sequences = [
                torch.from_numpy(record["sequence"])
                for _, record in batch
            ]

            lengths = torch.tensor(
                [len(sequence) for sequence in sequences],
                dtype=torch.long
            )

            # Pad sequences to the longest sequence in this batch.
            padded_sequences = pad_sequence(
                sequences,
                batch_first=True,
                padding_value=0.0
            ).to(device)

            # Generate logits and convert them into probabilities.
            logits = model(padded_sequences, lengths)
            probabilities = torch.sigmoid(logits).cpu().numpy()

            for (original_position, record), probability in zip(
                batch, probabilities
            ):
                probability = float(probability)

                predictions.append({
                    "original_position": original_position,
                    "user_id": record["user_id"],
                    "bot_probability": probability,
                    "predicted_label": int(probability >= threshold)
                })

    # Restore the original account order.
    predictions.sort(key=lambda item: item["original_position"])

    # Remove the internal ordering field.
    for prediction in predictions:
        del prediction["original_position"]

    return predictions

def predict_accounts(
    accounts,
    model=None,
    device=None,
    scaler_path=SCALER_PATH,
    batch_size=16,
    threshold=0.5,
):
    """
    Predict bot probability for one or more accounts provided
    as raw tweets.

    Each account must have the form:

    {
        "user_id": "...",
        "tweets": [
            {
                "tweet_id": "...",
                "created_at": "..."
            },
            ...
        ]
    }

    Returns:
        List of dictionaries:

        {
            "user_id": "...",
            "bot_probability": float,
            "predicted_label": int
        }
    """

    if not isinstance(accounts, list):
        raise ValueError("accounts must be provided as a list.")

    if not accounts:
        raise ValueError("accounts cannot be empty.")

    # Load the model only if the caller did not provide one.
    if model is None:
        model, device, _ = load_model(device=device)

    # If a model was provided without a device,
    # infer its current device from its parameters.
    elif device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            raise ValueError("The provided model has no parameters.")

    records = []

    for account in accounts:

        user_id = account.get("user_id") if isinstance(account, dict) else None

        if user_id is None or str(user_id).strip() == "":
            raise ValueError("Each account must have a valid user_id.")

        sequence = preprocess_account_tweets(
            account,
            scaler_path=scaler_path,
        )

        records.append({
            "user_id": str(user_id),
            "sequence": sequence,
        })

    return predict_records(
        records,
        model=model,
        device=device,
        batch_size=batch_size,
        threshold=threshold,
    )