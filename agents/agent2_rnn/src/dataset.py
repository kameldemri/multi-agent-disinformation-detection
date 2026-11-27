
import numpy as np
import torch

from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence

import torch


class AccountSequenceDataset(Dataset):
    """
    Dataset containing one temporal sequence per account.

    Each item contains:
        - sequence: [sequence_length, 5]
        - label: 0 (genuine) or 1 (bot)
    """

    def __init__(self, split):

        if split not in ["train", "validation", "test"]:
            raise ValueError(
                f"Invalid split: {split}"
            )

        from agents.agent2_rnn.src.preprocess import (
            PROCESSED_DATA_DIR
        )

        path = (
            PROCESSED_DATA_DIR /
            f"{split}_scaled_sequences.npz"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}"
            )

        print(f"Loading {split} dataset...")

        data = np.load(
            path,
            allow_pickle=True,
        )

        self.sequences = [
            np.asarray(sequence, dtype=np.float32)
            for sequence in data["sequences"]
        ]

        self.labels = np.asarray(
            data["labels"],
            dtype=np.int64,
        )

        data.close()

        if len(self.sequences) != len(self.labels):
            raise ValueError(
                "Sequences and labels have different lengths."
            )

        for i, sequence in enumerate(self.sequences):

            if sequence.ndim != 2 or sequence.shape[1] != 5:
                raise ValueError(
                    f"Invalid sequence shape at index {i}: "
                    f"{sequence.shape}"
                )

            if len(sequence) == 0:
                raise ValueError(
                    f"Empty sequence at index {i}"
                )

            if not np.isfinite(sequence).all():
                raise ValueError(
                    f"Non-finite values at index {i}"
                )

        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("Invalid labels detected.")

        self.lengths = np.asarray(
            [len(sequence) for sequence in self.sequences],
            dtype=np.int64,
        )

        print(f"  Accounts: {len(self.sequences):,}")
        print(f"  Total timesteps: {self.lengths.sum():,}")
        print(f"  Min length: {self.lengths.min():,}")
        print(f"  Max length: {self.lengths.max():,}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):

        sequence = torch.from_numpy(
            self.sequences[index]
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32,
        )

        return sequence, label


class LengthBasedBatchSampler(Sampler):
    """
    Groups accounts with similar sequence lengths.

    Training:
        Randomize indices, sort within large pools,
        create batches, then shuffle batch order.

    Evaluation:
        Deterministically sort all accounts by length.
    """

    def __init__(
        self,
        lengths,
        batch_size=8,
        shuffle=True,
        bucket_size_multiplier=10,
        seed=42,
    ):

        self.lengths = np.asarray(
            lengths,
            dtype=np.int64,
        )

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = (
            batch_size * bucket_size_multiplier
        )
        self.seed = seed
        self.epoch = 0

        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")

        if len(self.lengths) == 0:
            raise ValueError("No sequences provided.")

    def __iter__(self):

        rng = np.random.default_rng(
            self.seed + self.epoch
        )

        self.epoch += 1

        if self.shuffle:

            indices = rng.permutation(
                len(self.lengths)
            )

            batches = []

            # Sort only inside large pools.
            for start in range(
                0,
                len(indices),
                self.bucket_size,
            ):

                pool = indices[
                    start:start + self.bucket_size
                ]

                pool = sorted(
                    pool,
                    key=lambda idx: self.lengths[idx],
                    reverse=True,
                )

                # Create batches from similar-length accounts.
                for batch_start in range(
                    0,
                    len(pool),
                    self.batch_size,
                ):

                    batch = pool[
                        batch_start:batch_start + self.batch_size
                    ]

                    batches.append(batch)

            # Randomize the order of the batches.
            rng.shuffle(batches)

        else:

            indices = np.argsort(
                -self.lengths
            )

            batches = [
                indices[start:start + self.batch_size].tolist()
                for start in range(
                    0,
                    len(indices),
                    self.batch_size,
                )
            ]

        for batch in batches:
            yield batch

    def __len__(self):

        return (
            len(self.lengths) + self.batch_size - 1
        ) // self.batch_size


def collate_account_sequences(batch):
    """
    Dynamically pad a batch of account sequences.

    Sorts sequences by descending length because
    pack_padded_sequence will use enforce_sorted=True.
    """

    sequences, labels = zip(*batch)

    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long,
    )

    # Sort longest sequence first.
    sorted_indices = torch.argsort(
        lengths,
        descending=True,
    )

    sequences = [
        sequences[i]
        for i in sorted_indices
    ]

    labels = torch.stack([
        labels[i]
        for i in sorted_indices
    ])

    lengths = lengths[sorted_indices]

    # Pad only to the longest sequence in this batch.
    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0,
    )

    return (
        padded_sequences,
        labels,
        lengths,
    )


def validate_dataloader(
    split="train",
    batch_size=8
):
    """Validate the complete DataLoader and measure padding efficiency."""

    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pack_padded_sequence

    print(f"\nValidating complete {split} DataLoader...")

    dataset = AccountSequenceDataset(split)

    sampler = LengthBasedBatchSampler(
        dataset.lengths,
        batch_size=batch_size,
        shuffle=False
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_account_sequences
    )

    total_accounts = 0
    total_real_timesteps = 0
    total_padded_positions = 0
    total_batches = 0

    for sequences, labels, lengths in loader:

        assert sequences.ndim == 3
        assert sequences.shape[0] == len(lengths)
        assert sequences.shape[2] == 5
        assert torch.all(lengths[:-1] >= lengths[1:])
        assert torch.all((labels == 0) | (labels == 1))

        for i, length in enumerate(lengths):
            assert torch.all(sequences[i, length:, :] == 0)

        packed = pack_padded_sequence(
            sequences,
            lengths,
            batch_first=True,
            enforce_sorted=True
        )

        real_timesteps = int(lengths.sum())
        padded_positions = (
            sequences.shape[0] * sequences.shape[1]
            - real_timesteps
        )

        assert packed.data.shape[0] == real_timesteps

        total_accounts += len(lengths)
        total_real_timesteps += real_timesteps
        total_padded_positions += padded_positions
        total_batches += 1

    total_positions = total_real_timesteps + total_padded_positions

    padding_ratio = (
        total_padded_positions / total_positions
        if total_positions else 0
    )

    efficiency = 1 - padding_ratio

    print("\nCOMPLETE VALIDATION SUMMARY")
    print("Accounts:", total_accounts)
    print("Batches:", total_batches)
    print("Real timesteps:", total_real_timesteps)
    print("Padding positions:", total_padded_positions)
    print(f"Padding ratio: {padding_ratio:.4%}")
    print(f"Padding efficiency: {efficiency:.4%}")

    assert total_accounts == len(dataset)
    assert total_real_timesteps == int(dataset.lengths.sum())

    print("\nAll DataLoader validation checks passed.")



def benchmark_batching_efficiency(
    split="train",
    batch_size=8,
    multipliers=(10, 20, 40),
    seed=42
):
    """Compare random batching with length-based bucketing."""

    dataset = AccountSequenceDataset(split)
    lengths = dataset.lengths
    rng = np.random.default_rng(seed)

    def calculate_padding(batches):
        total_real = 0
        total_padded = 0

        for batch in batches:
            batch_lengths = lengths[batch]

            total_real += int(batch_lengths.sum())
            total_padded += (
                len(batch) * int(batch_lengths.max())
                - int(batch_lengths.sum())
            )

        total_positions = total_real + total_padded

        return {
            "padding": total_padded,
            "ratio": total_padded / total_positions,
            "efficiency": total_real / total_positions
        }

    # Random batching baseline
    indices = rng.permutation(len(lengths))

    random_batches = [
        indices[i:i + batch_size].tolist()
        for i in range(0, len(indices), batch_size)
    ]

    random_result = calculate_padding(random_batches)

    print("\nBATCHING EFFICIENCY COMPARISON")
    print(f"Random batching: {random_result['ratio']:.4%} padding")

    for multiplier in multipliers:

        sampler = LengthBasedBatchSampler(
            lengths,
            batch_size=batch_size,
            shuffle=True,
            bucket_size_multiplier=multiplier,
            seed=seed
        )

        batches = list(sampler)
        result = calculate_padding(batches)

        improvement = (
            (random_result["padding"] - result["padding"])
            / random_result["padding"] * 100
            if random_result["padding"] else 0
        )

        print(
            f"Bucket x{multiplier} "
            f"({batch_size * multiplier} accounts): "
            f"padding={result['ratio']:.4%}, "
            f"efficiency={result['efficiency']:.4%}, "
            f"padding reduction={improvement:.2f}%"
        )



def benchmark_batch_class_balance(
    split="train",
    batch_size=8,
    multipliers=(10, 20, 40),
    seeds=(42, 123, 456)
):
    """Measure class composition across length-based batches."""

    dataset = AccountSequenceDataset(split)
    labels = dataset.labels
    lengths = dataset.lengths

    print("\nBATCH CLASS BALANCE ANALYSIS")

    for multiplier in multipliers:

        print(f"\nBucket multiplier: {multiplier}")

        for seed in seeds:

            sampler = LengthBasedBatchSampler(
                lengths,
                batch_size=batch_size,
                shuffle=True,
                bucket_size_multiplier=multiplier,
                seed=seed
            )

            batch_bot_ratios = []
            single_class_batches = 0

            for batch in sampler:

                batch_labels = labels[batch]
                bot_ratio = float(batch_labels.mean())

                batch_bot_ratios.append(bot_ratio)

                if len(np.unique(batch_labels)) == 1:
                    single_class_batches += 1

            ratios = np.asarray(batch_bot_ratios)

            print(
                f"Seed {seed}: "
                f"mean bot ratio={ratios.mean():.3f}, "
                f"std={ratios.std():.3f}, "
                f"single-class batches={single_class_batches}/"
                f"{len(ratios)}"
            )



def analyze_length_label_relationship(split="train"):
    """Analyze the relationship between sequence length and account label."""

    dataset = AccountSequenceDataset(split)

    lengths = dataset.lengths
    labels = dataset.labels

    print("\nSEQUENCE LENGTH / LABEL ANALYSIS")

    for label, name in [(0, "Genuine"), (1, "Bot")]:
        class_lengths = lengths[labels == label]

        print(f"\n{name} accounts:")
        print("Count:", len(class_lengths))
        print("Mean length:", round(float(class_lengths.mean()), 2))
        print("Median length:", float(np.median(class_lengths)))
        print("Minimum:", int(class_lengths.min()))
        print("Maximum:", int(class_lengths.max()))

    # Divide all accounts into four equal-sized length groups.
    quartile_edges = np.quantile(
        lengths,
        [0, 0.25, 0.50, 0.75, 1.0]
    )

    print("\nBOT PROPORTION BY LENGTH QUARTILE")

    for i in range(4):
        if i < 3:
            mask = (
                (lengths >= quartile_edges[i])
                & (lengths < quartile_edges[i + 1])
            )
        else:
            mask = (
                (lengths >= quartile_edges[i])
                & (lengths <= quartile_edges[i + 1])
            )

        group_labels = labels[mask]

        print(
            f"Q{i + 1}: "
            f"accounts={len(group_labels)}, "
            f"bot_ratio={group_labels.mean():.3f}, "
            f"median_length={np.median(lengths[mask]):.1f}"
        )



def compare_length_label_across_splits():
    """Compare sequence-length and label distributions across splits."""

    print("\nLENGTH / LABEL DISTRIBUTION ACROSS SPLITS")

    for split in ["train", "validation", "test"]:

        dataset = AccountSequenceDataset(split)

        lengths = dataset.lengths
        labels = dataset.labels

        print(f"\n{'=' * 45}")
        print(f"SPLIT: {split.upper()}")
        print(f"{'=' * 45}")

        for label, name in [(0, "Genuine"), (1, "Bot")]:

            class_lengths = lengths[labels == label]

            print(
                f"{name}: "
                f"count={len(class_lengths)}, "
                f"mean={class_lengths.mean():.2f}, "
                f"median={np.median(class_lengths):.1f}"
            )

        # Use training-set quartile boundaries only.
        # This keeps the comparison consistent across splits.
        if split == "train":
            quartile_edges = np.quantile(
                lengths,
                [0, 0.25, 0.50, 0.75, 1.0]
            )

        print("Bot ratio by training-defined length quartiles:")

        for i in range(4):

            if i < 3:
                mask = (
                    (lengths >= quartile_edges[i])
                    & (lengths < quartile_edges[i + 1])
                )
            else:
                mask = (
                    (lengths >= quartile_edges[i])
                    & (lengths <= quartile_edges[i + 1])
                )

            group_labels = labels[mask]

            if len(group_labels) == 0:
                print(f"Q{i + 1}: no accounts")
                continue

            print(
                f"Q{i + 1}: "
                f"accounts={len(group_labels)}, "
                f"bot_ratio={group_labels.mean():.3f}"
            )



def create_dataloader(
    split,
    batch_size=8,
    bucket_size_multiplier=10,
    seed=42
):
    """
    Create a DataLoader for account-level temporal sequences.

    Training uses shuffled length-based buckets.
    Validation and test use deterministic length ordering.
    """

    from torch.utils.data import DataLoader

    if split not in ["train", "validation", "test"]:
        raise ValueError(
            "split must be train, validation, or test"
        )

    dataset = AccountSequenceDataset(split)

    sampler = LengthBasedBatchSampler(
        lengths=dataset.lengths,
        batch_size=batch_size,
        shuffle=(split == "train"),
        bucket_size_multiplier=bucket_size_multiplier,
        seed=seed
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_account_sequences
    )

    return loader
