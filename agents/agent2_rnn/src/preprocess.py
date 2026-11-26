
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

import sqlite3
from datetime import datetime, timezone

import numpy as np


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



def load_accounts_from_processed_file(file_path: Path) -> set[str]:
    """
    Read only user_id from a processed tweet file.
    Process in chunks to avoid loading millions of tweets into RAM.
    """
    account_ids = set()

    for chunk in pd.read_csv(
        file_path,
        usecols=["user_id"],
        dtype={"user_id": "string"},
        chunksize=CHUNK_SIZE,
    ):
        account_ids.update(chunk["user_id"].dropna().tolist())

    return account_ids


def create_account_splits():
    """
    Create reproducible train/validation/test assignments.
    Each account appears exactly once.
    """

    output_dir = PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    split_file = output_dir / "account_splits.csv"

    # Avoid accidentally replacing an existing experiment split.
    if split_file.exists():
        print(f"Account splits already exist: {split_file}")
        print("Keeping the existing assignments.")
        return

    genuine_file = output_dir / "genuine_tweets.csv"
    spambot_file = output_dir / "spambot_tweets.csv"

    if not genuine_file.exists() or not spambot_file.exists():
        raise FileNotFoundError(
            "Processed tweet files are missing. Run Phase 1 first."
        )

    print("\n" + "=" * 60)
    print("AGENT 2 - PHASE 2: ACCOUNT-LEVEL SPLITTING")
    print("=" * 60)

    # Recover only accounts that actually have usable tweets.
    genuine_ids = load_accounts_from_processed_file(genuine_file)
    spambot_ids = load_accounts_from_processed_file(spambot_file)

    # An account must not belong to both classes.
    overlap = genuine_ids.intersection(spambot_ids)

    if overlap:
        raise ValueError(
            f"Found {len(overlap)} account IDs in both classes."
        )

    print(f"Genuine accounts with tweets: {len(genuine_ids)}")
    print(f"Spambot accounts with tweets: {len(spambot_ids)}")

    # Build a small account-level dataset, not a tweet-level dataset.
    accounts = pd.DataFrame(
        [
            {"user_id": user_id, "label": 0}
            for user_id in sorted(genuine_ids)
        ]
        +
        [
            {"user_id": user_id, "label": 1}
            for user_id in sorted(spambot_ids)
        ]
    )

    # First split: 70% training, 30% temporary.
    train_accounts, temporary_accounts = train_test_split(
        accounts,
        test_size=0.30,
        random_state=42,
        stratify=accounts["label"],
    )

    # Second split: divide the temporary 30% equally.
    validation_accounts, test_accounts = train_test_split(
        temporary_accounts,
        test_size=0.50,
        random_state=42,
        stratify=temporary_accounts["label"],
    )

    # Attach the split name to each account.
    train_accounts = train_accounts.copy()
    validation_accounts = validation_accounts.copy()
    test_accounts = test_accounts.copy()

    train_accounts["split"] = "train"
    validation_accounts["split"] = "validation"
    test_accounts["split"] = "test"

    # Combine and save the assignments.
    splits = pd.concat(
        [train_accounts, validation_accounts, test_accounts],
        ignore_index=True,
    )

    splits = splits.sort_values("user_id").reset_index(drop=True)

    # Basic integrity checks.
    if splits["user_id"].duplicated().any():
        raise ValueError("An account appears in more than one split.")

    if len(splits) != len(accounts):
        raise ValueError("Some accounts are missing from the splits.")

    splits.to_csv(split_file, index=False)

    print("\nSplit sizes:")
    print(splits["split"].value_counts())

    print("\nClass distribution per split:")
    print(
        pd.crosstab(
            splits["split"],
            splits["label"],
            margins=True,
        )
    )

    print(f"\nTotal accounts: {len(splits)}")
    print(f"Saved assignments to: {split_file}")



def create_temporal_database():
    """
    Create the temporary SQLite database and its import tracking table.
    """

    db_path = PROCESSED_DATA_DIR / "temporal_processing.db"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS tweets (
                user_id TEXT NOT NULL,
                tweet_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                created_at_utc INTEGER NOT NULL,
                label INTEGER NOT NULL
            )
        """)

        # Prevent duplicate tweet IDs.
        connection.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_tweet_id
            ON tweets (tweet_id)
        """)

        # Track completed CSV imports.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS import_log (
                file_name TEXT PRIMARY KEY,
                label INTEGER NOT NULL,
                rows_read INTEGER NOT NULL,
                inserted INTEGER NOT NULL,
                invalid_timestamps INTEGER NOT NULL,
                invalid_rows INTEGER NOT NULL,
                completed_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Build the chronological index after importing the data.
        connection.execute("""
            DROP INDEX IF EXISTS idx_tweets_temporal
        """)

    print(f"SQLite database ready at: {db_path}")




def convert_twitter_timestamp(value: str) -> int:
    """
    Convert Twitter's original timestamp into UTC Unix seconds.
    Example: Wed Oct 10 20:19:24 +0000 2018
    """

    parsed = datetime.strptime(
        value,
        "%a %b %d %H:%M:%S %z %Y"
    )

    return int(parsed.timestamp())



def import_tweet_file_to_sqlite(
    connection: sqlite3.Connection,
    file_path: Path,
    label: int,
):
    """
    Import one processed tweet CSV into SQLite using chunks.
    """

    print(f"\nImporting: {file_path.name}")

    total_read = 0
    total_inserted = 0
    invalid_timestamps = 0
    invalid_rows = 0

    insert_query = """
        INSERT OR IGNORE INTO tweets (
            user_id,
            tweet_id,
            created_at,
            created_at_utc,
            label
        )
        VALUES (?, ?, ?, ?, ?)
    """

    for chunk in pd.read_csv(
        file_path,
        usecols=["user_id", "id", "created_at"],
        dtype={
            "user_id": "string",
            "id": "string",
            "created_at": "string",
        },
        chunksize=CHUNK_SIZE,
    ):

        total_read += len(chunk)
        records = []

        for row in chunk.itertuples(index=False):

            user_id = row.user_id
            tweet_id = row.id
            created_at = row.created_at

            if pd.isna(user_id) or pd.isna(tweet_id) or pd.isna(created_at):
                invalid_rows += 1
                continue

            try:
                timestamp_utc = convert_twitter_timestamp(created_at)
            except (ValueError, TypeError, OverflowError):
                invalid_timestamps += 1
                continue

            records.append(
                (
                    str(user_id),
                    str(tweet_id),
                    str(created_at),
                    timestamp_utc,
                    label,
                )
            )

        if records:
            before = connection.total_changes

            connection.executemany(insert_query, records)
            connection.commit()

            after = connection.total_changes
            total_inserted += after - before

        print(
            f"Processed {total_read:,} rows | "
            f"Inserted {total_inserted:,} tweets"
        )

    print(f"Finished: {file_path.name}")
    print(f"Rows read: {total_read:,}")
    print(f"Tweets inserted: {total_inserted:,}")
    print(f"Invalid timestamps: {invalid_timestamps:,}")
    print(f"Invalid rows: {invalid_rows:,}")

    return {
        "read": total_read,
        "inserted": total_inserted,
        "invalid_timestamps": invalid_timestamps,
        "invalid_rows": invalid_rows,
    }



def import_all_tweets_to_sqlite():
    """
    Import both tweet datasets into SQLite.
    Completed files are skipped on subsequent runs.
    """

    create_temporal_database()

    db_path = PROCESSED_DATA_DIR / "temporal_processing.db"

    files = [
        (PROCESSED_DATA_DIR / "genuine_tweets.csv", 0),
        (PROCESSED_DATA_DIR / "spambot_tweets.csv", 1),
    ]

    with sqlite3.connect(db_path) as connection:

        for file_path, label in files:

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Missing processed file: {file_path}"
                )

            completed = connection.execute(
                """
                SELECT 1 FROM import_log
                WHERE file_name = ?
                """,
                (file_path.name,),
            ).fetchone()

            if completed:
                print(f"\nAlready imported: {file_path.name}")
                continue

            stats = import_tweet_file_to_sqlite(
                connection,
                file_path,
                label,
            )

            connection.execute(
                """
                INSERT INTO import_log (
                    file_name,
                    label,
                    rows_read,
                    inserted,
                    invalid_timestamps,
                    invalid_rows
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    file_path.name,
                    label,
                    stats["read"],
                    stats["inserted"],
                    stats["invalid_timestamps"],
                    stats["invalid_rows"],
                ),
            )

            connection.commit()

        # Create the chronological index after importing.
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_tweets_temporal
            ON tweets (user_id, created_at_utc, tweet_id)
        """)

        print("\n" + "=" * 55)
        print("SQLITE IMPORT SUMMARY")
        print("=" * 55)

        for label, count in connection.execute(
            """
            SELECT label, COUNT(*)
            FROM tweets
            GROUP BY label
            ORDER BY label
            """
        ):
            class_name = "Genuine" if label == 0 else "Spambot"
            print(f"{class_name}: {count:,} tweets")

        total = connection.execute(
            "SELECT COUNT(*) FROM tweets"
        ).fetchone()[0]

        print(f"\nTotal tweets in database: {total:,}")



def verify_temporal_database():
    """
    Verify SQLite tweet records against the account-level splits.
    """

    db_path = PROCESSED_DATA_DIR / "temporal_processing.db"
    split_path = PROCESSED_DATA_DIR / "account_splits.csv"

    splits = pd.read_csv(
        split_path,
        dtype={"user_id": "string"},
    )

    with sqlite3.connect(db_path) as connection:

        # Check imported tweet counts.
        print("\nTweet counts by class:")

        counts = pd.read_sql_query(
            """
            SELECT label, COUNT(*) AS tweet_count
            FROM tweets
            GROUP BY label
            ORDER BY label
            """,
            connection,
        )

        print(counts.to_string(index=False))

        # Get one record per account.
        accounts = pd.read_sql_query(
            """
            SELECT
                user_id,
                MIN(label) AS min_label,
                MAX(label) AS max_label,
                COUNT(*) AS tweet_count
            FROM tweets
            GROUP BY user_id
            """,
            connection,
            dtype={"user_id": "string"},
        )

        # Check that each account has only one label.
        inconsistent_labels = accounts[
            accounts["min_label"] != accounts["max_label"]
        ]

        # Compare SQLite accounts with account_splits.csv.
        comparison = accounts.merge(
            splits[["user_id", "label", "split"]],
            on="user_id",
            how="outer",
            indicator=True,
        )

        missing_accounts = comparison[
            comparison["_merge"] != "both"
        ]

        label_mismatches = comparison[
            (comparison["_merge"] == "both")
            & (comparison["min_label"] != comparison["label"])
        ]

        # Check timestamp range.
        timestamp_range = connection.execute(
            """
            SELECT
                MIN(created_at_utc),
                MAX(created_at_utc)
            FROM tweets
            """
        ).fetchone()

        # Check completed imports.
        imports = pd.read_sql_query(
            "SELECT * FROM import_log",
            connection,
        )

    print("\nAccount verification:")
    print(f"Accounts in SQLite: {len(accounts):,}")
    print(f"Accounts in split CSV: {len(splits):,}")
    print(f"Inconsistent account labels: {len(inconsistent_labels)}")
    print(f"Missing or unexpected accounts: {len(missing_accounts)}")
    print(f"Label mismatches: {len(label_mismatches)}")

    print("\nTimestamp range (UTC Unix seconds):")
    print(timestamp_range)

    print("\nImport log:")
    print(imports.to_string(index=False))

    if (
        len(inconsistent_labels) > 0
        or len(missing_accounts) > 0
        or len(label_mismatches) > 0
        or len(accounts) != 2074
        or len(imports) != 2
    ):
        raise ValueError("Database integrity verification failed.")

    print("\nSUCCESS: Database integrity checks passed.")



def export_temporal_features():
    """
    Calculate temporal features for every tweet and export them to CSV.

    Processing is performed account by account to avoid loading
    the complete dataset into memory.
    """

    db_path = PROCESSED_DATA_DIR / "temporal_processing.db"
    split_path = PROCESSED_DATA_DIR / "account_splits.csv"
    output_path = PROCESSED_DATA_DIR / "temporal_features.csv"

    if not db_path.exists():
        raise FileNotFoundError("SQLite database not found.")

    if not split_path.exists():
        raise FileNotFoundError("Account splits file not found.")

    # Load the small account-level split mapping.
    splits = pd.read_csv(
        split_path,
        dtype={"user_id": "string"},
    )

    split_mapping = dict(
        zip(splits["user_id"], splits["split"])
    )

    # Avoid accidentally appending to an old output.
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. "
            "Rename or remove it before regenerating features."
        )

    query = """
        SELECT
            user_id,
            tweet_id,
            created_at_utc,
            label
        FROM tweets
        ORDER BY user_id, created_at_utc, tweet_id
    """

    output_columns = [
        "user_id",
        "tweet_id",
        "gap_seconds",
        "hour",
        "day_of_week",
        "label",
        "split",
    ]

    total_processed = 0
    current_user_id = None
    previous_timestamp = None

    buffer = []
    BUFFER_SIZE = 10_000

    with sqlite3.connect(db_path) as connection:

        cursor = connection.execute(query)

        with open(output_path, "w", encoding="utf-8", newline="") as output_file:

            # Write the CSV header.
            pd.DataFrame(columns=output_columns).to_csv(
                output_file,
                index=False,
            )

            while True:

                rows = cursor.fetchmany(CHUNK_SIZE)

                if not rows:
                    break

                for user_id, tweet_id, timestamp, label in rows:

                    if user_id != current_user_id:
                        current_user_id = user_id
                        previous_timestamp = None

                    # First tweet of an account has no previous tweet.
                    if previous_timestamp is None:
                        gap_seconds = 0
                    else:
                        gap_seconds = timestamp - previous_timestamp

                    # Convert UTC Unix timestamp into datetime.
                    tweet_datetime = datetime.fromtimestamp(
                        timestamp,
                        tz=datetime.now().astimezone().tzinfo,
                    )

                    # Convert to UTC explicitly.
                    tweet_datetime = datetime.fromtimestamp(
                        timestamp,
                        tz=timezone.utc,
                    )

                    hour = tweet_datetime.hour
                    day_of_week = tweet_datetime.weekday()

                    buffer.append(
                        (
                            user_id,
                            tweet_id,
                            gap_seconds,
                            hour,
                            day_of_week,
                            label,
                            split_mapping[user_id],
                        )
                    )

                    previous_timestamp = timestamp
                    total_processed += 1

                if len(buffer) >= BUFFER_SIZE:

                    pd.DataFrame(
                        buffer,
                        columns=output_columns,
                    ).to_csv(
                        output_file,
                        index=False,
                        header=False,
                    )

                    buffer.clear()

                    print(
                        f"Processed {total_processed:,} tweets"
                    )

            # Write remaining records.
            if buffer:
                pd.DataFrame(
                    buffer,
                    columns=output_columns,
                ).to_csv(
                    output_file,
                    index=False,
                    header=False,
                )

    print("\n" + "=" * 55)
    print("TEMPORAL FEATURE EXTRACTION COMPLETED")
    print("=" * 55)
    print(f"Total tweets processed: {total_processed:,}")
    print(f"Output file: {output_path}")


def verify_temporal_features():
    """
    Validate temporal_features.csv in chunks.
    """

    output_path = PROCESSED_DATA_DIR / "temporal_features.csv"
    split_path = PROCESSED_DATA_DIR / "account_splits.csv"

    expected_splits = pd.read_csv(
        split_path,
        dtype={"user_id": "string"},
    )

    expected = {
        row.user_id: (int(row.label), row.split)
        for row in expected_splits.itertuples(index=False)
    }

    seen_accounts = set()
    previous_user = None
    total_rows = 0

    for chunk in pd.read_csv(
        output_path,
        dtype={
            "user_id": "string",
            "tweet_id": "string",
            "split": "string",
        },
        chunksize=CHUNK_SIZE,
    ):

        if chunk.isna().any().any():
            raise ValueError("Missing values found in temporal features.")

        if (chunk["gap_seconds"] < 0).any():
            raise ValueError("Negative time gaps detected.")

        if not chunk["hour"].between(0, 23).all():
            raise ValueError("Invalid hour values detected.")

        if not chunk["day_of_week"].between(0, 6).all():
            raise ValueError("Invalid weekday values detected.")

        # Detect account transitions across all rows and chunks.
        for row in chunk[["user_id", "gap_seconds"]].itertuples(index=False):

            if row.user_id != previous_user:

                if row.user_id in seen_accounts:
                    raise ValueError(
                        f"Account appears in multiple separate groups: {row.user_id}"
                    )

                seen_accounts.add(row.user_id)

                if row.gap_seconds != 0:
                    raise ValueError(
                        f"First tweet has a nonzero gap: {row.user_id}"
                    )

            previous_user = row.user_id

        # Check account labels and splits within this chunk.
        grouped = chunk.groupby("user_id", sort=False).agg(
            label_count=("label", "nunique"),
            split_count=("split", "nunique"),
            label=("label", "first"),
            split=("split", "first"),
        )

        for user_id, row in grouped.iterrows():

            if user_id not in expected:
                raise ValueError(f"Unknown account: {user_id}")

            expected_label, expected_split = expected[user_id]

            if (
                row.label_count != 1
                or row.split_count != 1
                or row.label != expected_label
                or row.split != expected_split
            ):
                raise ValueError(
                    f"Incorrect label or split for account: {user_id}"
                )

        total_rows += len(chunk)
        
        print(f"Validated {total_rows:,} tweets")

    if total_rows != 4_449_395:
        raise ValueError(f"Unexpected row count: {total_rows:,}")

    if seen_accounts != set(expected):
        missing_accounts = set(expected) - seen_accounts
        extra_accounts = seen_accounts - set(expected)

        print("\nACCOUNT COVERAGE MISMATCH")
        print(f"Expected accounts: {len(expected)}")
        print(f"Found accounts: {len(seen_accounts)}")
        print(f"Missing accounts: {len(missing_accounts)}")
        print(f"Extra accounts: {len(extra_accounts)}")

        print("\nFirst 10 missing account IDs:")
        print(list(missing_accounts)[:10])

        print("\nFirst 10 extra account IDs:")
        print(list(extra_accounts)[:10])

        raise ValueError("Account coverage does not match account_splits.csv.")

    print("\n" + "=" * 55)
    print("TEMPORAL FEATURE VALIDATION")
    print("=" * 55)
    print(f"Total tweets: {total_rows:,}")
    print(f"Distinct accounts: {len(seen_accounts):,}")
    print("Negative gaps: 0")
    print("Invalid hours: 0")
    print("Invalid weekdays: 0")
    print("Incorrect first-tweet gaps: 0")
    print("Label/split mismatches: 0")
    print("\nSUCCESS: All temporal feature checks passed.")


def analyze_sequence_lengths():
    """
    Analyze the number of temporal observations (tweets)
    available for each account.
    """

    output_path = PROCESSED_DATA_DIR / "temporal_features.csv"

    if not output_path.exists():
        raise FileNotFoundError(
            f"Temporal features file not found: {output_path}"
        )

    account_counts = {}

    for chunk in pd.read_csv(
        output_path,
        usecols=["user_id"],
        dtype={"user_id": "string"},
        chunksize=CHUNK_SIZE,
    ):
        counts = chunk["user_id"].value_counts()

        for user_id, count in counts.items():
            account_counts[user_id] = (
                account_counts.get(user_id, 0) + int(count)
            )

        print(
            f"Processed {sum(account_counts.values()):,} tweets"
        )

    lengths = pd.Series(account_counts, dtype="int64")

    print("\n" + "=" * 55)
    print("SEQUENCE LENGTH ANALYSIS")
    print("=" * 55)

    print(f"Accounts: {len(lengths):,}")
    print(f"Total tweets: {lengths.sum():,}")
    print(f"Minimum: {lengths.min():,}")
    print(f"Maximum: {lengths.max():,}")

    print("\nPercentiles:")

    percentiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    for p in percentiles:
        print(
            f"P{int(p * 100):02d}: "
            f"{lengths.quantile(p):,.1f}"
        )

    print("\nAccounts by sequence length:")

    limits = [
        (1, 10),
        (11, 50),
        (51, 100),
        (101, 500),
        (501, 1000),
        (1001, 2000),
        (2001, 5000),
        (5001, float("inf")),
    ]

    for lower, upper in limits:
        count = ((lengths >= lower) & (lengths <= upper)).sum()

        if upper == float("inf"):
            label = f"{lower:,}+"
        else:
            label = f"{lower:,}-{upper:,}"

        print(f"{label}: {count:,}")

    print("\nSUCCESS: Sequence length analysis completed.")


    #sequence construction : 
def build_account_sequences():
    """
    Build one variable-length temporal sequence per account.

    Each timestep contains:
        [gap_seconds, hour, day_of_week]

    Sequences are kept variable-length at this stage.
    No scaling, padding, truncation, or feature transformation
    is performed here.
    """

    input_path = PROCESSED_DATA_DIR / "temporal_features.csv"
    split_path = PROCESSED_DATA_DIR / "account_splits.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Temporal features file not found: {input_path}"
        )

    if not split_path.exists():
        raise FileNotFoundError(
            f"Account split file not found: {split_path}"
        )

    # ---------------------------------------------------------
    # Load account -> (label, split) mapping.
    # This file contains only 2,074 rows, so it is safe to load.
    # ---------------------------------------------------------

    split_df = pd.read_csv(
        split_path,
        dtype={
            "user_id": "string",
            "label": "int8",
            "split": "string",
        },
    )

    expected_accounts = set(split_df["user_id"])

    account_info = {
        row.user_id: (int(row.label), row.split)
        for row in split_df.itertuples(index=False)
    }

    # ---------------------------------------------------------
    # Prepare containers for the three splits.
    #
    # We deliberately keep sequences variable-length.
    # Therefore each split is a list of NumPy arrays.
    # ---------------------------------------------------------

    sequences = {
        "train": [],
        "validation": [],
        "test": [],
    }

    labels = {
        "train": [],
        "validation": [],
        "test": [],
    }

    current_user = None
    current_sequence = []

    processed_rows = 0
    processed_accounts = 0

    def finalize_account(user_id, sequence):
        """Store the completed sequence for one account."""

        if user_id is None:
            return

        if user_id not in account_info:
            raise ValueError(
                f"Account {user_id} not found in account_splits.csv."
            )

        label, split = account_info[user_id]

        if len(sequence) == 0:
            raise ValueError(
                f"Account {user_id} has an empty sequence."
            )

        sequence_array = np.asarray(
            sequence,
            dtype=np.float32,
        )

        sequences[split].append(sequence_array)
        labels[split].append(label)

    # ---------------------------------------------------------
    # Stream the temporal features CSV.
    # ---------------------------------------------------------

    for chunk in pd.read_csv(
        input_path,
        usecols=[
            "user_id",
            "gap_seconds",
            "hour",
            "day_of_week",
        ],
        dtype={
            "user_id": "string",
            "gap_seconds": "float32",
            "hour": "int8",
            "day_of_week": "int8",
        },
        chunksize=CHUNK_SIZE,
    ):

        for row in chunk.itertuples(index=False):

            user_id = row.user_id

            # First account.
            if current_user is None:
                current_user = user_id

            # Account changed.
            elif user_id != current_user:

                finalize_account(
                    current_user,
                    current_sequence,
                )

                processed_accounts += 1

                if processed_accounts % 100 == 0:
                    print(
                        f"Constructed {processed_accounts:,} "
                        f"account sequences"
                    )

                current_user = user_id
                current_sequence = []

            # Add current timestep.
            current_sequence.append(
                [
                    float(row.gap_seconds),
                    int(row.hour),
                    int(row.day_of_week),
                ]
            )

            processed_rows += 1

    # Final account after the loop.
    if current_user is not None:
        finalize_account(
            current_user,
            current_sequence,
        )
        processed_accounts += 1

    # ---------------------------------------------------------
    # Validate account counts.
    # ---------------------------------------------------------

    total_sequences = sum(
        len(sequences[split])
        for split in sequences
    )

    if total_sequences != len(expected_accounts):
        raise ValueError(
            "Sequence count does not match account_splits.csv: "
            f"{total_sequences} != {len(expected_accounts)}"
        )

    if processed_rows != 4_449_395:
        raise ValueError(
            f"Unexpected number of processed rows: {processed_rows}"
        )

    # ---------------------------------------------------------
    # Save each split.
    #
    # object arrays are required because sequences have different
    # lengths at this stage.
    # ---------------------------------------------------------

    for split in ["train", "validation", "test"]:

        output_path = (
            PROCESSED_DATA_DIR /
            f"{split}_sequences.npz"
        )

        if output_path.exists():
            raise FileExistsError(
                f"Output already exists: {output_path}"
            )

        sequence_array = np.asarray(
            sequences[split],
            dtype=object,
        )

        label_array = np.asarray(
            labels[split],
            dtype=np.int8,
        )

        np.savez_compressed(
            output_path,
            sequences=sequence_array,
            labels=label_array,
        )

        print(
            f"\nSaved {split}:"
        )
        print(
            f"  Accounts: {len(sequence_array):,}"
        )
        print(
            f"  File: {output_path}"
        )

    print("\n" + "=" * 55)
    print("ACCOUNT SEQUENCE CONSTRUCTION")
    print("=" * 55)
    print(f"Processed tweets: {processed_rows:,}")
    print(f"Constructed accounts: {processed_accounts:,}")

    for split in ["train", "validation", "test"]:
        print(
            f"{split.capitalize():12}: "
            f"{len(sequences[split]):,} sequences"
        )

    print("\nSUCCESS: Account sequences constructed.")

    #validate the constructed sequences
def verify_account_sequences():
    """
    Validate the account-level sequence files created by
    build_account_sequences().
    """

    expected_counts = {
        "train": 1451,
        "validation": 311,
        "test": 312,
    }

    expected_total_tweets = 4_449_395

    total_accounts = 0
    total_timesteps = 0

    print("\n" + "=" * 55)
    print("ACCOUNT SEQUENCE VALIDATION")
    print("=" * 55)

    for split in ["train", "validation", "test"]:

        path = (
            PROCESSED_DATA_DIR /
            f"{split}_sequences.npz"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Sequence file not found: {path}"
            )

        data = np.load(
            path,
            allow_pickle=True,
        )

        sequences = data["sequences"]
        labels = data["labels"]

        expected_accounts = expected_counts[split]

        if len(sequences) != expected_accounts:
            raise ValueError(
                f"{split}: unexpected account count: "
                f"{len(sequences)} != {expected_accounts}"
            )

        if len(labels) != expected_accounts:
            raise ValueError(
                f"{split}: sequence/label count mismatch."
            )

        split_timesteps = 0

        for i, sequence in enumerate(sequences):

            if sequence.ndim != 2:
                raise ValueError(
                    f"{split}: sequence {i} is not 2D: "
                    f"shape={sequence.shape}"
                )

            if sequence.shape[1] != 3:
                raise ValueError(
                    f"{split}: sequence {i} does not have "
                    f"3 features: shape={sequence.shape}"
                )

            if sequence.shape[0] == 0:
                raise ValueError(
                    f"{split}: sequence {i} is empty."
                )

            if np.isnan(sequence).any():
                raise ValueError(
                    f"{split}: sequence {i} contains NaN values."
                )

            if labels[i] not in [0, 1]:
                raise ValueError(
                    f"{split}: invalid label at index {i}: "
                    f"{labels[i]}"
                )

            split_timesteps += sequence.shape[0]

        total_accounts += len(sequences)
        total_timesteps += split_timesteps

        print(f"\n{split.upper()}")
        print(f"Accounts: {len(sequences):,}")
        print(f"Timesteps: {split_timesteps:,}")
        print(
            f"Label 0: {(labels == 0).sum():,}"
        )
        print(
            f"Label 1: {(labels == 1).sum():,}"
        )
        print(
            f"Min sequence length: "
            f"{min(seq.shape[0] for seq in sequences):,}"
        )
        print(
            f"Max sequence length: "
            f"{max(seq.shape[0] for seq in sequences):,}"
        )

        data.close()

    print("\n" + "-" * 55)
    print(f"Total accounts: {total_accounts:,}")
    print(f"Total timesteps: {total_timesteps:,}")

    if total_accounts != 2074:
        raise ValueError(
            f"Unexpected total account count: {total_accounts}"
        )

    if total_timesteps != expected_total_tweets:
        raise ValueError(
            f"Unexpected total timestep count: "
            f"{total_timesteps:,} != "
            f"{expected_total_tweets:,}"
        )

    print("\nSUCCESS: All account sequence checks passed.")

def analyze_raw_sequence_features():
    """
    Inspect the raw feature ranges stored in account sequences.
    """

    print("\n" + "=" * 55)
    print("RAW SEQUENCE FEATURE ANALYSIS")
    print("=" * 55)

    for split in ["train", "validation", "test"]:

        path = (
            PROCESSED_DATA_DIR /
            f"{split}_sequences.npz"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Sequence file not found: {path}"
            )

        data = np.load(
            path,
            allow_pickle=True,
        )

        sequences = data["sequences"]

        all_gaps = np.concatenate(
            [seq[:, 0] for seq in sequences]
        )

        hours = np.concatenate(
            [seq[:, 1] for seq in sequences]
        )

        weekdays = np.concatenate(
            [seq[:, 2] for seq in sequences]
        )

        print(f"\n{split.upper()}")

        print(
            f"Gap min:    {all_gaps.min():,.0f}"
        )
        print(
            f"Gap median: {np.median(all_gaps):,.0f}"
        )
        print(
            f"Gap mean:   {all_gaps.mean():,.2f}"
        )
        print(
            f"Gap max:    {all_gaps.max():,.0f}"
        )

        print(
            f"Hour range: "
            f"{hours.min():.0f} - {hours.max():.0f}"
        )

        print(
            f"Weekday range: "
            f"{weekdays.min():.0f} - {weekdays.max():.0f}"
        )

        data.close()

    print("\nSUCCESS: Raw sequence feature analysis completed.")


def transform_temporal_sequences():
    """
    Transform raw temporal features into RNN-ready temporal features.

    Raw features:
        [gap_seconds, hour, day_of_week]

    Transformed features:
        [log_gap, hour_sin, hour_cos, day_sin, day_cos]
    """

    print("\n" + "=" * 55)
    print("TRANSFORMING TEMPORAL SEQUENCES")
    print("=" * 55)

    for split in ["train", "validation", "test"]:

        input_path = (
            PROCESSED_DATA_DIR /
            f"{split}_sequences.npz"
        )

        output_path = (
            PROCESSED_DATA_DIR /
            f"{split}_rnn_sequences.npz"
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input sequence file not found: {input_path}"
            )

        print(f"\nProcessing {split}...")

        data = np.load(
            input_path,
            allow_pickle=True,
        )

        raw_sequences = data["sequences"]

        transformed_sequences = []

        for sequence in raw_sequences:

            gap_seconds = sequence[:, 0]
            hour = sequence[:, 1]
            day_of_week = sequence[:, 2]

            # 1. Compress the highly skewed time gaps
            log_gap = np.log1p(gap_seconds)

            # 2. Cyclic representation of hour
            hour_sin = np.sin(
                2 * np.pi * hour / 24
            )

            hour_cos = np.cos(
                2 * np.pi * hour / 24
            )

            # 3. Cyclic representation of weekday
            day_sin = np.sin(
                2 * np.pi * day_of_week / 7
            )

            day_cos = np.cos(
                2 * np.pi * day_of_week / 7
            )

            transformed = np.column_stack([
                log_gap,
                hour_sin,
                hour_cos,
                day_sin,
                day_cos,
            ]).astype(np.float32)

            transformed_sequences.append(transformed)

        labels = data["labels"]

        np.savez_compressed(
            output_path,
            sequences=np.array(
                transformed_sequences,
                dtype=object,
            ),
            labels=labels,
        )

        data.close()

        print(
            f"  Sequences: {len(transformed_sequences):,}"
        )
        print(
            f"  Features per timestep: "
            f"{transformed_sequences[0].shape[1]}"
        )

        print(
            f"  Saved: {output_path.name}"
        )

    print(
        "\nSUCCESS: Temporal sequence transformation completed."
    )

def verify_transformed_sequences():
    """
    Verify the transformed RNN-ready temporal sequences.
    """

    print("\n" + "=" * 55)
    print("VERIFYING TRANSFORMED SEQUENCES")
    print("=" * 55)

    expected_accounts = {
        "train": 1451,
        "validation": 311,
        "test": 312,
    }

    total_accounts = 0

    for split in ["train", "validation", "test"]:

        path = (
            PROCESSED_DATA_DIR /
            f"{split}_rnn_sequences.npz"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Transformed sequence file not found: {path}"
            )

        data = np.load(
            path,
            allow_pickle=True,
        )

        sequences = data["sequences"]
        labels = data["labels"]

        if len(sequences) != expected_accounts[split]:
            raise ValueError(
                f"{split}: account count mismatch"
            )

        if len(labels) != len(sequences):
            raise ValueError(
                f"{split}: labels/sequences mismatch"
            )

        all_values = np.concatenate(sequences, axis=0)

        if all_values.shape[1] != 5:
            raise ValueError(
                f"{split}: expected 5 features, "
                f"found {all_values.shape[1]}"
            )

        if np.isnan(all_values).any():
            raise ValueError(
                f"{split}: NaN detected"
            )

        if np.isinf(all_values).any():
            raise ValueError(
                f"{split}: Inf detected"
            )

        log_gap = all_values[:, 0]
        hour_sin = all_values[:, 1]
        hour_cos = all_values[:, 2]
        day_sin = all_values[:, 3]
        day_cos = all_values[:, 4]

        # log1p(gap) must never be negative
        if log_gap.min() < 0:
            raise ValueError(
                f"{split}: negative log_gap detected"
            )

        # Cyclic features must remain in [-1, 1]
        cyclic_features = np.concatenate([
            hour_sin,
            hour_cos,
            day_sin,
            day_cos,
        ])

        if cyclic_features.min() < -1.000001:
            raise ValueError(
                f"{split}: cyclic feature below -1"
            )

        if cyclic_features.max() > 1.000001:
            raise ValueError(
                f"{split}: cyclic feature above 1"
            )

        total_accounts += len(sequences)

        print(f"\n{split.upper()}")
        print(f"  Accounts: {len(sequences):,}")
        print(f"  Total timesteps: {len(all_values):,}")

        print(
            f"  log_gap range: "
            f"{log_gap.min():.4f} - {log_gap.max():.4f}"
        )

        print(
            f"  hour_sin range: "
            f"{hour_sin.min():.4f} - {hour_sin.max():.4f}"
        )

        print(
            f"  hour_cos range: "
            f"{hour_cos.min():.4f} - {hour_cos.max():.4f}"
        )

        print(
            f"  day_sin range: "
            f"{day_sin.min():.4f} - {day_sin.max():.4f}"
        )

        print(
            f"  day_cos range: "
            f"{day_cos.min():.4f} - {day_cos.max():.4f}"
        )

        data.close()

    if total_accounts != 2074:
        raise ValueError(
            f"Total account count mismatch: {total_accounts}"
        )

    print(
        "\nSUCCESS: Transformed sequences are valid."
    )

def fit_temporal_scaler():
    """
    Compute feature-wise mean and standard deviation using
    TRAIN sequences only.

    Saves the scaler parameters for later use.
    """

    print("\n" + "=" * 55)
    print("FITTING TEMPORAL SCALER")
    print("=" * 55)

    train_path = (
        PROCESSED_DATA_DIR /
        "train_rnn_sequences.npz"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training sequences not found: {train_path}"
        )

    data = np.load(
        train_path,
        allow_pickle=True,
    )

    sequences = data["sequences"]

    # Online statistics
    feature_sum = np.zeros(5, dtype=np.float64)
    feature_sum_squared = np.zeros(5, dtype=np.float64)
    total_count = 0

    for sequence in sequences:

        sequence = sequence.astype(
            np.float64,
            copy=False,
        )

        feature_sum += sequence.sum(axis=0)

        feature_sum_squared += (
            np.square(sequence).sum(axis=0)
        )

        total_count += sequence.shape[0]

    data.close()

    mean = feature_sum / total_count

    variance = (
        feature_sum_squared / total_count
        - np.square(mean)
    )

    # Numerical protection against tiny negative values
    variance = np.maximum(variance, 0.0)

    std = np.sqrt(variance)

    if np.any(std == 0):
        raise ValueError(
            "At least one feature has zero variance."
        )

    scaler_path = (
        PROCESSED_DATA_DIR /
        "temporal_scaler.npz"
    )

    np.savez(
        scaler_path,
        mean=mean,
        std=std,
    )

    print(f"\nTraining timesteps: {total_count:,}")

    print("\nFeature statistics:")

    feature_names = [
        "log_gap",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ]

    for name, m, s in zip(
        feature_names,
        mean,
        std,
    ):
        print(
            f"  {name:<10} "
            f"mean={m:.6f}  "
            f"std={s:.6f}"
        )

    print(
        f"\nSaved scaler: {scaler_path.name}"
    )

    print(
        "\nSUCCESS: Temporal scaler fitted on TRAIN only."
    )


def apply_temporal_scaler():
    """
    Apply the train-fitted scaler to train, validation and test.

    The scaler parameters are never recalculated here.
    """

    print("\n" + "=" * 55)
    print("APPLYING TEMPORAL SCALER")
    print("=" * 55)

    scaler_path = (
        PROCESSED_DATA_DIR /
        "temporal_scaler.npz"
    )

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}"
        )

    scaler = np.load(scaler_path)

    mean = scaler["mean"]
    std = scaler["std"]

    scaler.close()

    for split in ["train", "validation", "test"]:

        input_path = (
            PROCESSED_DATA_DIR /
            f"{split}_rnn_sequences.npz"
        )

        output_path = (
            PROCESSED_DATA_DIR /
            f"{split}_scaled_sequences.npz"
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input not found: {input_path}"
            )

        print(f"\nProcessing {split}...")

        data = np.load(
            input_path,
            allow_pickle=True,
        )

        raw_sequences = data["sequences"]
        labels = data["labels"]

        scaled_sequences = []

        for sequence in raw_sequences:

            scaled = (
                sequence.astype(np.float32) - mean
            ) / std

            scaled_sequences.append(
                scaled.astype(np.float32)
            )

        np.savez_compressed(
            output_path,
            sequences=np.array(
                scaled_sequences,
                dtype=object,
            ),
            labels=labels,
        )

        print(f"  Accounts: {len(scaled_sequences):,}")
        print(f"  Features: 5")
        print(f"  Saved: {output_path.name}")

        data.close()

    print(
        "\nSUCCESS: Scaler applied to all splits."
    )


def verify_scaled_sequences():
    """
    Verify scaled sequences and calculate feature statistics.

    Statistics are accumulated incrementally to avoid
    concatenating all timesteps into a large matrix.
    """

    print("\n" + "=" * 55)
    print("VERIFYING SCALED SEQUENCES")
    print("=" * 55)

    expected_accounts = {
        "train": 1451,
        "validation": 311,
        "test": 312,
    }

    feature_names = [
        "log_gap",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ]

    total_accounts = 0
    total_timesteps = 0

    for split in ["train", "validation", "test"]:

        path = (
            PROCESSED_DATA_DIR /
            f"{split}_scaled_sequences.npz"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Scaled sequences not found: {path}"
            )

        data = np.load(path, allow_pickle=True)

        sequences = data["sequences"]
        labels = data["labels"]

        if len(sequences) != expected_accounts[split]:
            raise ValueError(
                f"{split}: incorrect account count"
            )

        if len(labels) != len(sequences):
            raise ValueError(
                f"{split}: labels/sequences mismatch"
            )

        if not np.isin(labels, [0, 1]).all():
            raise ValueError(
                f"{split}: invalid labels detected"
            )

        feature_sum = np.zeros(5, dtype=np.float64)
        feature_sum_squared = np.zeros(5, dtype=np.float64)
        count = 0

        for i, sequence in enumerate(sequences):

            if sequence.ndim != 2 or sequence.shape[1] != 5:
                raise ValueError(
                    f"{split}: invalid shape in sequence {i}"
                )

            if len(sequence) == 0:
                raise ValueError(
                    f"{split}: empty sequence {i}"
                )

            if not np.isfinite(sequence).all():
                raise ValueError(
                    f"{split}: NaN or Inf in sequence {i}"
                )

            sequence = sequence.astype(
                np.float64,
                copy=False,
            )

            feature_sum += sequence.sum(axis=0)
            feature_sum_squared += np.square(sequence).sum(axis=0)
            count += len(sequence)

        means = feature_sum / count

        variances = (
            feature_sum_squared / count
            - np.square(means)
        )

        variances = np.maximum(variances, 0)
        stds = np.sqrt(variances)

        print(f"\n{split.upper()}")
        print(f"  Accounts: {len(sequences):,}")
        print(f"  Timesteps: {count:,}")
        print(f"  Labels: {np.unique(labels)}")

        print("  Feature statistics:")

        for name, mean, std in zip(
            feature_names,
            means,
            stds,
        ):
            print(
                f"    {name:<10} "
                f"mean={mean:.5f}  std={std:.5f}"
            )

        total_accounts += len(sequences)
        total_timesteps += count

        data.close()

    if total_accounts != 2074:
        raise ValueError(
            f"Expected 2074 accounts, found {total_accounts}"
        )

    if total_timesteps != 4_449_395:
        raise ValueError(
            f"Unexpected total timesteps: {total_timesteps}"
        )

    print("\nTOTAL")
    print(f"  Accounts: {total_accounts:,}")
    print(f"  Timesteps: {total_timesteps:,}")

    print(
        "\nSUCCESS: Scaled sequences are valid."
    )
