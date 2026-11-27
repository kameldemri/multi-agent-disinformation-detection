import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from agents.agent2_rnn.src.dataset import create_dataloader
from agents.agent2_rnn.src.model import AccountBiGRU


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHECKPOINT_DIR = (
    PROJECT_ROOT / "checkpoints" / "agent2"
)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

CONFIG = {
    "input_size": 5,
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.3,

    "batch_size": 16,
    "bucket_size_multiplier": 10,

    "learning_rate": 0.001,
    "weight_decay": 0.01,

    "max_epochs": 30,
    "patience": 5,

    "gradient_clip": 1.0,

    "threshold": 0.5,
    "seed": 42,
}


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one complete epoch.

    Returns:
        average_loss: Mean loss across accounts.
    """

    model.train()

    total_loss = 0.0
    total_accounts = 0

    for sequences, labels, lengths in dataloader:

        # Move tensors to the selected device
        sequences = sequences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        # Reset gradients from the previous batch
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        logits = model(sequences, lengths)

        # Calculate classification loss
        loss = criterion(logits, labels)

        # Backpropagation
        loss.backward()

        # Prevent excessively large gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=CONFIG["gradient_clip"]
        )

        # Update model parameters
        optimizer.step()

        # Accumulate loss weighted by batch size
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_accounts += batch_size

    average_loss = total_loss / total_accounts

    return average_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model without updating its parameters.

    Returns:
        Dictionary containing loss and classification metrics.
    """

    model.eval()

    total_loss = 0.0
    total_accounts = 0

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for sequences, labels, lengths in dataloader:

            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            # Forward pass
            logits = model(sequences, lengths)

            # Validation loss
            loss = criterion(logits, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_accounts += batch_size

            # Convert logits into probabilities
            probabilities = torch.sigmoid(logits)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    average_loss = total_loss / total_accounts

    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Convert probabilities into binary predictions
    predictions = (
        all_probabilities >= CONFIG["threshold"]
    ).astype(int)

    metrics = {
        "loss": average_loss,

        "accuracy": accuracy_score(
            all_labels, predictions
        ),

        "precision": precision_score(
            all_labels, predictions, zero_division=0
        ),

        "recall": recall_score(
            all_labels, predictions, zero_division=0
        ),

        "f1": f1_score(
            all_labels, predictions, zero_division=0
        ),

        "roc_auc": roc_auc_score(
            all_labels, all_probabilities
        ),
    }

    return metrics



def save_checkpoint(
    model,
    optimizer,
    epoch,
    validation_metrics,
    path,
):
    """Save model and optimizer states."""

    # Convert NumPy scalar values into native Python types.
    safe_metrics = {
        key: float(value)
        for key, value in validation_metrics.items()
    }

    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": safe_metrics,
        "config": CONFIG.copy(),
    }

    torch.save(checkpoint, path)



class EarlyStopping:
    """Stop training when validation ROC-AUC stops improving."""

    def __init__(self, patience=5, min_delta=0.0):

        self.patience = patience
        self.min_delta = min_delta

        self.best_score = float("-inf")
        self.counter = 0

    def step(self, score):

        if score > self.best_score + self.min_delta:

            self.best_score = score
            self.counter = 0

            return True, False

        self.counter += 1

        should_stop = self.counter >= self.patience

        return False, should_stop


def train_model(model, train_loader, validation_loader, device):
    """
    Complete training controller.

    Trains the model, evaluates validation performance,
    saves the best checkpoint, and applies early stopping.
    """

    criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    early_stopping = EarlyStopping(
        patience=CONFIG["patience"]
    )

    best_checkpoint_path = CHECKPOINT_DIR / "best_bigru.pt"

    history_path = CHECKPOINT_DIR / "training_history.csv"

    history = []

    print("\nStarting training...")
    print(f"Device: {device}")
    print(f"Maximum epochs: {CONFIG['max_epochs']}")

    for epoch in range(1, CONFIG["max_epochs"] + 1):

        # ------------------------------
        # Training
        # ------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        # ------------------------------
        # Validation
        # ------------------------------

        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        validation_auc = validation_metrics["roc_auc"]

        if not np.isfinite(validation_auc):
            raise ValueError(
                f"Validation ROC-AUC is not finite: {validation_auc}"
            )

        # ------------------------------
        # Early stopping
        # ------------------------------

        improved, should_stop = early_stopping.step(
            validation_auc
        )

        # ------------------------------
        # Save best checkpoint
        # ------------------------------

        if improved:

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_metrics=validation_metrics,
                path=best_checkpoint_path,
            )

        # ------------------------------
        # Record history
        # ------------------------------

        epoch_results = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{
                f"val_{key}": value
                for key, value in validation_metrics.items()
            },
        }

        history.append(epoch_results)
        with history_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=epoch_results.keys(),
            )
            writer.writeheader()
            writer.writerows(history)

        # ------------------------------
        # Display progress
        # ------------------------------

        print(
            f"\nEpoch [{epoch}/{CONFIG['max_epochs']}]"
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {validation_metrics['loss']:.4f}")
        print(f"Accuracy: {validation_metrics['accuracy']:.4f}")
        print(f"Precision: {validation_metrics['precision']:.4f}")
        print(f"Recall: {validation_metrics['recall']:.4f}")
        print(f"F1: {validation_metrics['f1']:.4f}")
        print(f"ROC-AUC: {validation_auc:.4f}")

        if improved:
            print("New best model saved.")

        if should_stop:
            print(
                f"\nEarly stopping triggered at epoch {epoch}."
            )
            break

    # Restore the model weights from the best validation checkpoint
    if not best_checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint was saved at: {best_checkpoint_path}"
    )

    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Restored best model from epoch {checkpoint['epoch']}")

    print("\nTraining completed.")
    print(f"Best validation ROC-AUC: {early_stopping.best_score:.4f}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Training history: {history_path}")

    return history


if __name__ == "__main__":

    set_seed(CONFIG["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading training data...")

    train_loader = create_dataloader(
        split="train",
        batch_size=CONFIG["batch_size"],
        bucket_size_multiplier=CONFIG["bucket_size_multiplier"],
        seed=CONFIG["seed"],
    )

    print("\nLoading validation data...")

    validation_loader = create_dataloader(
        split="validation",
        batch_size=CONFIG["batch_size"],
        bucket_size_multiplier=CONFIG["bucket_size_multiplier"],
        seed=CONFIG["seed"],
    )

    model = AccountBiGRU(
        input_size=CONFIG["input_size"],
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    ).to(device)

    print("\nModel initialized.")
    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")

    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
    )
