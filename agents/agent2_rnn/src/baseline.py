
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from agents.agent2_rnn.src.dataset import AccountSequenceDataset

from sklearn.ensemble import RandomForestClassifier


def run_length_baseline():

    print("=" * 60)
    print("ACTIVITY-VOLUME BASELINE")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------

    print("\nLoading training dataset...")
    train_dataset = AccountSequenceDataset("train")

    print("\nLoading validation dataset...")
    val_dataset = AccountSequenceDataset("validation")

    # --------------------------------------------------
    # 2. Extract the only feature: account sequence length
    # --------------------------------------------------

    X_train = np.log1p(train_dataset.lengths).reshape(-1, 1)
    y_train = train_dataset.labels

    X_val = np.log1p(val_dataset.lengths).reshape(-1, 1)
    y_val = val_dataset.labels

    print("\nDataset information:")
    print(f"Training accounts:   {len(y_train)}")
    print(f"Validation accounts: {len(y_val)}")
    print("Number of features:  1 (log1p(sequence length))")

    # --------------------------------------------------
    # 3. Build the model
    # --------------------------------------------------

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    # --------------------------------------------------
    # 4. Train using TRAIN only
    # --------------------------------------------------

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    print("Training completed.")

    # --------------------------------------------------
    # 5. Evaluate on validation data
    # --------------------------------------------------

    print("\nEvaluating on validation data...")

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_prob)

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        y_pred,
        target_names=["Genuine", "Spambot"],
        zero_division=0
    ))

    return model


def run_random_forest_baseline():

    print("\n" + "=" * 60)
    print("RANDOM FOREST ACTIVITY-VOLUME BASELINE")
    print("=" * 60)

    # 1. Load datasets

    print("\nLoading training dataset...")
    train_dataset = AccountSequenceDataset("train")

    print("\nLoading validation dataset...")
    val_dataset = AccountSequenceDataset("validation")

    # 2. Extract the same feature used by Baseline 1

    X_train = np.log1p(train_dataset.lengths).reshape(-1, 1)
    y_train = train_dataset.labels

    X_val = np.log1p(val_dataset.lengths).reshape(-1, 1)
    y_val = val_dataset.labels

    # 3. Build Random Forest

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        class_weight=None,
        random_state=42,
        n_jobs=-1
    )

    # 4. Train

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    print("Training completed.")

    # Evaluate on training data to inspect overfitting

    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]

    print("\nTraining performance:")
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"F1-score: {f1_score(y_train, y_train_pred):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_train, y_train_prob):.4f}")

    # 5. Predict

    print("\nEvaluating on validation data...")

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    # 6. Metrics

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_prob)

    print("\n" + "=" * 60)
    print("RANDOM FOREST VALIDATION RESULTS")
    print("=" * 60)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        y_pred,
        target_names=["Genuine", "Spambot"],
        zero_division=0
    ))

    return model


if __name__ == "__main__":
    run_length_baseline()
    run_random_forest_baseline()

