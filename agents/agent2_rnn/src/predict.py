from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from agents.agent2_rnn.src.model import AccountBiGRU

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "agent2"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "agent2"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "best_bigru.pt"
DEFAULT_SCALER = PROCESSED_DATA_DIR / "temporal_scaler.npz"

_MODEL_CACHE: dict[tuple[str, str, str], tuple] = {}


def load_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    scaler_path: Path = DEFAULT_SCALER,
    device: str | None = None,
):
    checkpoint_path = Path(checkpoint_path)
    scaler_path = Path(scaler_path)
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cache_key = (
        str(checkpoint_path),
        str(scaler_path),
        str(selected_device),
    )

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=selected_device,
        weights_only=True,
    )
    config = dict(checkpoint.get("config", {}))

    model = AccountBiGRU(
        input_size=config.get("input_size", 5),
        hidden_size=config.get("hidden_size", 64),
        num_layers=config.get("num_layers", 1),
        dropout=config.get("dropout", 0.3),
    ).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with np.load(scaler_path) as scaler:
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        std = np.asarray(scaler["std"], dtype=np.float32)

    if mean.shape != (5,) or std.shape != (5,):
        raise ValueError("Scaler must contain two arrays of length 5.")

    result = (model, mean, std, config, selected_device)
    _MODEL_CACHE[cache_key] = result
    return result


def prepare_sequence(sequence, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float64)

    if array.ndim != 2:
        raise ValueError("Sequence must be a 2D array.")

    if array.shape[1] == 3:
        gap_seconds = array[:, 0]
        hour = array[:, 1]
        day_of_week = array[:, 2]
        array = np.column_stack(
            [
                np.log1p(gap_seconds),
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * day_of_week / 7),
                np.cos(2 * np.pi * day_of_week / 7),
            ]
        )
    elif array.shape[1] != 5:
        raise ValueError("Sequence must have 3 raw features or 5 transformed features.")

    array = array.astype(np.float32)

    if not np.isfinite(array).all():
        raise ValueError("Sequence contains NaN or infinite values.")

    if len(array) == 0:
        raise ValueError("Sequence is empty.")

    return ((array - mean) / std).astype(np.float32)


def score_accounts(
    sequences,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    scaler_path: Path = DEFAULT_SCALER,
    device: str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    model, mean, std, _, selected_device = load_model(
        checkpoint_path,
        scaler_path,
        device,
    )

    prepared = [prepare_sequence(sequence, mean, std) for sequence in sequences]

    if not prepared:
        return np.asarray([], dtype=np.float64)

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    order = sorted(
        range(len(prepared)),
        key=lambda index: len(prepared[index]),
        reverse=True,
    )
    scores = np.empty(len(prepared), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            batch_indices = order[start:start + batch_size]
            tensors = [
                torch.from_numpy(prepared[index])
                for index in batch_indices
            ]
            lengths = torch.tensor(
                [tensor.shape[0] for tensor in tensors],
                dtype=torch.long,
                device=selected_device,
            )
            padded = pad_sequence(
                tensors,
                batch_first=True,
                padding_value=0.0,
            ).to(selected_device)
            logits = model(padded, lengths)
            probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float64)

            for index, probability in zip(batch_indices, probabilities):
                scores[index] = float(probability)

    return scores


def score_account(
    sequence,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    scaler_path: Path = DEFAULT_SCALER,
    device: str | None = None,
) -> float:
    return float(
        score_accounts(
            [sequence],
            checkpoint_path=checkpoint_path,
            scaler_path=scaler_path,
            device=device,
        )[0]
    )


def score_split(
    split: str,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    scaler_path: Path = DEFAULT_SCALER,
    device: str | None = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    if split not in ["train", "validation", "test"]:
        raise ValueError("split must be train, validation, or test.")

    data_path = PROCESSED_DATA_DIR / f"{split}_scaled_sequences.npz"
    split_path = PROCESSED_DATA_DIR / "account_splits.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Sequence file not found: {data_path}")

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with np.load(data_path, allow_pickle=True) as data:
        sequences = list(data["sequences"])
        stored_labels = np.asarray(data["labels"], dtype=np.int64)

    accounts = pd.read_csv(
        split_path,
        dtype={"user_id": "string", "split": "string"},
    )
    accounts = accounts[accounts["split"] == split]
    accounts = accounts.sort_values("user_id").reset_index(drop=True)

    if len(accounts) != len(sequences):
        raise ValueError(
            f"Split has {len(accounts)} accounts but {len(sequences)} sequences."
        )

    if len(stored_labels) != len(accounts):
        raise ValueError("Stored labels do not match the split file.")

    expected_labels = accounts["label"].to_numpy(dtype=np.int64)

    if not np.array_equal(stored_labels, expected_labels):
        raise ValueError("Stored labels do not match account_splits.csv.")

    scores = score_accounts(
        sequences,
        checkpoint_path=checkpoint_path,
        scaler_path=scaler_path,
        device=device,
        batch_size=batch_size,
    )

    return pd.DataFrame(
        {
            "user_id": accounts["user_id"].astype(str),
            "score": scores,
            "label": expected_labels,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score accounts with Agent 2.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--split", choices=["train", "validation", "test"])
    group.add_argument("--sequences-json")
    group.add_argument("--sequences-file", type=Path)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--scaler", default=str(DEFAULT_SCALER))
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    if args.split:
        results = score_split(
            args.split,
            checkpoint_path=args.checkpoint,
            scaler_path=args.scaler,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        raw_json = (
            args.sequences_json
            if args.sequences_json is not None
            else args.sequences_file.read_text(encoding="utf-8")
        )
        payload = json.loads(raw_json)
        sequences = payload["sequences"] if isinstance(payload, dict) else [payload]
        scores = score_accounts(
            sequences,
            checkpoint_path=args.checkpoint,
            scaler_path=args.scaler,
            device=args.device,
            batch_size=args.batch_size,
        )
        results = pd.DataFrame({"score": scores})

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"Saved {len(results)} scores to {args.output}")
    else:
        print(results.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
