
from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cresci_2017"

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed" / "agent2"
)

CHUNK_SIZE = 50_000

CLASSES = {
    0: "genuine_accounts",
    1: "social_spambots",
}

REQUIRED_TWEET_COLUMNS = [
    "user_id",
    "id",
    "created_at",
]


# ============================================================
# 2. LOAD ACCOUNT IDS
# ============================================================

def load_user_ids(folder_name: str) -> set[str]:
    """
    Load account IDs from users.csv.

    Only the ID column is loaded to minimize memory usage.
    """

    users_path = RAW_DATA_DIR / folder_name / "users.csv"

    users = pd.read_csv(
        users_path,
        usecols=["id"],
        dtype={"id": "string"},
    )

    user_ids = set(users["id"].dropna())

    print(f"  Accounts found: {len(user_ids):,}")

    return user_ids


# ============================================================
# 3. PROCESS TWEETS IN CHUNKS
# ============================================================

def process_tweet_file(
    folder_name: str,
    label: int,
    valid_user_ids: set[str],
) -> dict:
    """
    Read a tweet CSV in chunks, validate records,
    and write the retained records directly to disk.
    """

    tweets_path = RAW_DATA_DIR / folder_name / "tweets.csv"

    output_path = (
        PROCESSED_DATA_DIR
        / f"{'genuine' if label == 0 else 'spambot'}_tweets.csv"
    )

    print(f"\nProcessing: {folder_name}")

    # Remove an old output to prevent accidental appending.
    if output_path.exists():
        output_path.unlink()

    stats = {
        "rows_read": 0,
        "rows_saved": 0,
        "missing_user_id": 0,
        "unknown_user_id": 0,
        "missing_created_at": 0,
        "unique_users": set(),
    }

    first_chunk = True

    reader = pd.read_csv(
        tweets_path,
        usecols=REQUIRED_TWEET_COLUMNS,
        dtype={
            "user_id": "string",
            "id": "string",
            "created_at": "string",
        },
        chunksize=CHUNK_SIZE,
        encoding="latin-1",
    )

    for chunk_number, chunk in enumerate(reader, start=1):

        stats["rows_read"] += len(chunk)

        # Remove records without an account ID.
        missing_user = chunk["user_id"].isna()
        stats["missing_user_id"] += int(missing_user.sum())

        chunk = chunk.loc[~missing_user].copy()

        # Keep only tweets belonging to known accounts.
        known_user = chunk["user_id"].isin(valid_user_ids)

        stats["unknown_user_id"] += int((~known_user).sum())

        chunk = chunk.loc[known_user].copy()

        # Remove records without a publication timestamp.
        missing_time = chunk["created_at"].isna()

        stats["missing_created_at"] += int(missing_time.sum())

        chunk = chunk.loc[~missing_time].copy()

        # Add the account-level label.
        chunk["label"] = label

        # Write this chunk directly to disk.
        chunk.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        first_chunk = False

        stats["rows_saved"] += len(chunk)

        stats["unique_users"].update(chunk["user_id"].unique())

        print(
            f"  Chunk {chunk_number}: "
            f"{len(chunk):,} valid tweets",
            flush=True,
        )

    return stats


# ============================================================
# 4. COMPLETE PHASE 1
# ============================================================

def prepare_tweet_data() -> None:
    """
    Prepare both classes and save their filtered tweets.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_users = set()

    print("=" * 60)
    print("AGENT 2 - PHASE 1: DATA PREPARATION")
    print("=" * 60)

    for label, folder_name in CLASSES.items():

        # Load the account IDs.
        valid_user_ids = load_user_ids(folder_name)

        # Process the corresponding tweets.
        stats = process_tweet_file(
            folder_name=folder_name,
            label=label,
            valid_user_ids=valid_user_ids,
        )

        total_users.update(stats["unique_users"])

        print("\nClass summary:")
        print(f"  Rows read          : {stats['rows_read']:,}")
        print(f"  Rows saved         : {stats['rows_saved']:,}")
        print(f"  Missing user IDs   : {stats['missing_user_id']:,}")
        print(f"  Unknown user IDs   : {stats['unknown_user_id']:,}")
        print(f"  Missing timestamps : {stats['missing_created_at']:,}")
        print(f"  Users with tweets  : {len(stats['unique_users']):,}")

    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)

    print(f"Total usable accounts: {len(total_users):,}")
    print(f"Output directory: {PROCESSED_DATA_DIR}")


# ============================================================
# 5. EXECUTION
# ============================================================

if __name__ == "__main__":
    prepare_tweet_data()
