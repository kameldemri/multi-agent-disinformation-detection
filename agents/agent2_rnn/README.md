# Agent 2 — Temporal Bot Detection (BiGRU)

## 1. Overview

Agent 2 is the temporal behavior analysis component of the multi-agent bot and disinformation detection system.

It uses a Bidirectional Gated Recurrent Unit (BiGRU) neural network to analyze the temporal activity patterns of social media accounts and identify behavior associated with automated accounts (bots).

Unlike content-based approaches, Agent 2 focuses on the chronological patterns of an account's tweets, including the time intervals between consecutive tweets and their publication times.

## 2. Objective

The objective of Agent 2 is to estimate the likelihood that a social media account exhibits bot-like temporal behavior.

For each account, the model produces:

- A bot probability representing the estimated likelihood of bot behavior.
- A predicted binary label: `1` for bot and `0` for genuine account.

Agent 2 provides behavioral evidence to the overall multi-agent system. Its output can subsequently be combined with the results of other agents to support the final detection and moderation decision.

## 3. Dataset and Labeling

Agent 2 was developed using the **Cresci-2017 dataset**, which contains genuine and automated Twitter accounts.

The dataset was organized at the account level. Each account is represented by its chronological sequence of tweets.

### 3.1 Account Labels

The binary classification labels are:

| Label | Account Type    |
| ----- | --------------- |
| `0`   | Genuine account |
| `1`   | Social spam bot |

Each tweet inherits the label of the account to which it belongs.

### 3.2 Dataset Statistics

After removing one tweet record with a missing user_id from the genuine-account data, the dataset contains:

| Category         | Number of accounts |
| ---------------- | ------------------ |
| Genuine accounts | 1,083              |
| Social spam bots | 991                |
| **Total**        | **2,074**          |

A total of **4,449,395 usable tweets** were retained.

### 3.3 Account-Level Data Splitting

The dataset was divided into training, validation, and test sets at the account level.

| Split      |   Genuine |    Bots |     Total |
| ---------- | --------: | ------: | --------: |
| Training   |       758 |     693 |     1,451 |
| Validation |       162 |     149 |       311 |
| Test       |       163 |     149 |       312 |
| **Total**  | **1,083** | **991** | **2,074** |

Splitting by account ensures that tweets belonging to the same account cannot appear in different splits.

The random seed was fixed to `42` to ensure reproducibility.

## 4. Temporal Feature Engineering

Agent 2 does not use tweet text as input. Instead, it extracts temporal information from each account's tweet history.

### 4.1 Timestamp Processing

For each tweet, the preprocessing pipeline extracts its creation timestamp from `created_at`.

The timestamps are:

- Parsed and converted to UTC.
- Sorted chronologically.
- Ordered by tweet ID when timestamps are identical.

The time difference between consecutive tweets is then calculated in seconds.

The first tweet of each account has a time gap of `0`.

### 4.2 Temporal Features

Each tweet is represented by five numerical features:

| Feature    | Description                                                                            |
| ---------- | -------------------------------------------------------------------------------------- |
| `log_gap`  | Log-transformed time interval between consecutive tweets , using `log1p(gap_seconds)`. |
| `hour_sin` | Sine encoding of the UTC hour of publication.                                          |
| `hour_cos` | Cosine encoding of the UTC hour of publication.                                        |
| `day_sin`  | Sine encoding of the day of the week.                                                  |
| `day_cos`  | Cosine encoding of the day of the week.                                                |

Cyclical encoding represents periodic time information while preserving the relationship between values near the boundaries of a cycle.

For example, 23:00 and 00:00 are close in time, even though their numerical hour values are far apart.

### 4.3 Feature Scaling

The five features are standardized using the mean and standard deviation calculated from the training data.

The same training scaler is reused during validation, testing, and inference.

This prevents information from validation or test data from influencing the scaling parameters.

The resulting representation for each account is a chronological sequence with shape:

`[number_of_tweets, 5]`

## 5. BiGRU Model Architecture

Agent 2 uses a Bidirectional Gated Recurrent Unit (BiGRU) to learn temporal patterns from each account's tweet sequence.

### 5.1 Architecture Configuration

The model uses the following configuration:

| Parameter                  |  Value |
| -------------------------- | -----: |
| Input size                 |      5 |
| Hidden size                |     64 |
| Number of GRU layers       |      1 |
| Bidirectional              |    Yes |
| Dropout                    |    0.3 |
| Total trainable parameters | 27,393 |

### 5.2 Network Structure

The model follows this processing flow:

1. **Input sequence:** Receives the five temporal features for every tweet.
2. **Packed sequences:** Processes variable-length account sequences while avoiding unnecessary computation on padded positions.
3. **Bidirectional GRU:** Processes the sequence in both forward and backward directions to capture temporal dependencies from both directions.
4. **Hidden-state concatenation:** Combines the final forward and backward hidden representations, producing a 128-dimensional representation.
5. **Dropout:** Applies dropout with a probability of 0.3 to reduce overfitting.
6. **Fully connected layer:** Maps the 128-dimensional representation to a single classification logit.

### 5.3 Variable-Length Sequences

Accounts have different numbers of tweets. Therefore, sequences are padded within each batch to allow efficient batch processing.

A length-based batching strategy groups accounts with similar sequence lengths to reduce unnecessary padding.

The model uses packed sequences through PyTorch's `pack_padded_sequence`, ensuring that padded positions are not treated as real tweet observations.

### 5.4 Classification Output

The final layer produces one logit per account.

A sigmoid function converts this logit into a probability between 0 and 1:

- Probability closer to `1`: stronger bot-like behavioral signal.
- Probability closer to `0`: stronger genuine-account behavioral signal.

The default classification threshold is `0.5`.

## 6. Training Configuration and Evaluation Results

### 6.1 Training Configuration

The BiGRU model was trained using PyTorch on a Google Colab environment with an NVIDIA Tesla T4 GPU.

The main training configuration was:

| Parameter                |                            Value |
| ------------------------ | -------------------------------: |
| Optimizer                |                            AdamW |
| Learning rate            |                            0.001 |
| Weight decay             |                             0.01 |
| Loss function            | Binary Cross-Entropy with Logits |
| Batch size               |                               16 |
| Maximum epochs           |                               30 |
| Early stopping patience  |                                5 |
| Gradient clipping        |                              1.0 |
| Classification threshold |                              0.5 |
| Random seed              |                               42 |

The best checkpoint was selected based on validation performance.

### 6.2 Validation Results

The best model checkpoint was obtained at epoch 8.

| Metric    | Validation |
| --------- | ---------: |
| Loss      |     0.0870 |
| Accuracy  |     96.78% |
| Precision |    100.00% |
| Recall    |     93.29% |
| F1-score  |     96.53% |
| ROC-AUC   |     0.9944 |

### 6.3 Test Results

The final evaluation on the held-out test set produced:

| Metric    |    Test |
| --------- | ------: |
| Loss      |  0.0658 |
| Accuracy  |  98.08% |
| Precision | 100.00% |
| Recall    |  95.97% |
| F1-score  |  97.95% |
| ROC-AUC   |  0.9964 |

The test confusion matrix was:

|                | Predicted Genuine | Predicted Bot |
| -------------- | ----------------: | ------------: |
| Actual Genuine |               163 |             0 |
| Actual Bot     |                 6 |           143 |

The model correctly classified all 163 genuine accounts and identified 143 of the 149 bot accounts.

### 6.4 Activity-Volume Baseline Comparison

To evaluate whether the BiGRU learned more than account activity volume alone, it was compared with a Random Forest baseline using only the number of tweets per account.

| Metric    | Random Forest |   BiGRU |
| --------- | ------------: | ------: |
| Accuracy  |        86.86% |  98.08% |
| Precision |        91.54% | 100.00% |
| Recall    |        79.87% |  95.97% |
| F1-score  |        85.30% |  97.95% |
| ROC-AUC   |        0.9047 |  0.9964 |

The BiGRU outperformed the activity-volume baseline across all reported metrics, suggesting that its temporal sequence representation provides useful information beyond tweet volume alone.

### 6.5 Temporal-Order Ablation

An additional experiment evaluated the trained model on test sequences whose tweet order was shuffled.

The resulting ROC-AUC was 0.9117, compared with 0.9964 on the original chronological sequences.

This indicates that the model's predictions are sensitive to the original temporal representation. However, shuffling also disrupts the relationship between consecutive timestamps and their time gaps, so this experiment does not isolate the causal contribution of chronological order alone.

## 7. Inference Pipeline

Agent 2 provides an inference pipeline that accepts raw tweet data for social media accounts and returns account-level bot predictions.

### 7.1 Inference Workflow

The inference process follows these steps:

1. Receive an account ID and its collection of raw tweets.
2. Validate the input data and required fields.
3. Parse tweet creation timestamps and convert them to UTC.
4. Sort tweets chronologically.
5. Extract the five temporal features for each tweet.
6. Apply the same feature transformations used during training.
7. Standardize the features using the saved training scaler.
8. Use the trained BiGRU model, loading the checkpoint if no model is supplied.
9. Convert the output logit into a bot probability using the sigmoid function.
10. Apply the classification threshold to obtain the predicted label.

### 7.2 Minimum Input Requirements

Each account must contain:

- A non-empty `user_id`.
- At least three tweets.
- A valid `tweet_id` for each tweet.
- A valid `created_at` timestamp for each tweet.

Supported timestamp formats include:

- Twitter format: `Wed Oct 10 20:19:24 +0000 2018`
- ISO 8601: `2018-10-10T20:19:24Z`
- ISO 8601 with timezone offset: `2018-10-10T21:19:24+01:00`

Timezone information is required. Timezone-naive timestamps are rejected.

### 7.3 Model and Scaler Artifacts

Inference requires two saved artifacts:

| Artifact                                    | Purpose                             |
| ------------------------------------------- | ----------------------------------- |
| `artifacts/best_bigru_safe.pt`              | Trained BiGRU checkpoint            |
| `data/processed/agent2/temporal_scaler.npz` | Training feature-scaling parameters |

The scaler contains the mean and standard deviation calculated exclusively from the training features. These saved parameters are reused during inference without refitting.

The checkpoint contains the trained model weights and configuration. The scaler contains the mean and standard deviation calculated from the training features.

Both artifacts must be available at their expected project-relative paths.

## 8. Input Format

The main inference function is `predict_accounts()`, which accepts a list of account dictionaries.

Each account must contain:

- `user_id`: the account identifier.
- `tweets`: a list of tweet dictionaries.

Each tweet must contain:

- `tweet_id`: the tweet identifier.
- `created_at`: the tweet creation timestamp.

### 8.1 Example Input

```python
accounts = [
    {
        "user_id": "123456789",
        "tweets": [
            {
                "tweet_id": "101",
                "created_at": "Wed Oct 10 20:19:24 +0000 2018"
            },
            {
                "tweet_id": "102",
                "created_at": "2018-10-10T21:00:00Z"
            },
            {
                "tweet_id": "103",
                "created_at": "2018-10-10T22:00:00+01:00"
            }
        ]
    }
]
```

The example illustrates the expected input structure and supported timestamp formats. Actual predictions require real collected tweet data.

### 8.2 Input Considerations

- Tweets must belong to the account identified by `user_id`.
- Each account must contain at least three tweets.
- Tweet text and engagement metrics are not required by Agent 2.
- Tweets do not need to be supplied in chronological order because the preprocessing pipeline sorts them.
- All timestamps must include timezone information.

The input should contain the collected tweet history available for each account, rather than manually calculated temporal features.

## 9. Output Contract and Agent 3 Integration

Agent 2 returns one prediction for each input account.

Each prediction is represented as a dictionary containing three fields:

| Field             | Type    | Description                                                        |
| ----------------- | ------- | ------------------------------------------------------------------ |
| `user_id`         | String  | Identifier of the analyzed account.                                |
| `bot_probability` | Float   | Model-estimated probability of bot-like behavior, between 0 and 1. |
| `predicted_label` | Integer | Binary prediction: `0` for genuine or `1` for bot.                 |

### 9.1 Example Output

```python
[
    {
        "user_id": "123456789",
        "bot_probability": 0.93,
        "predicted_label": 1
    },
    {
        "user_id": "987654321",
        "bot_probability": 0.08,
        "predicted_label": 0
    }
]
```

### 9.2 Interpretation

- A higher `bot_probability` indicates stronger evidence of bot-like temporal behavior.
- A lower `bot_probability` indicates stronger evidence of genuine-account behavior.
- `predicted_label` is obtained by comparing the probability with the configured classification threshold, which defaults to `0.5`.

### 9.3 Integration with Agent 3

Agent 2 provides behavioral evidence to Agent 3, which can combine it with outputs from other agents to calculate or determine the system's final risk assessment.

The `bot_probability` must not be interpreted as the probability that an account spreads misinformation.

It represents only the BiGRU model's estimate of bot-like behavior based on temporal activity patterns.

The output contract is independent of Agent 3's final decision-making algorithm.

## 10. Usage Example

The following example demonstrates how to load the trained model and generate predictions from raw account data.

Run the code from the project root directory.

```python
from agents.agent2_rnn.src.predict import (
    load_model,
    predict_accounts
)

# Load the trained BiGRU model
model, device, _ = load_model()
# Prepare raw account data
accounts = [
    {
        "user_id": "123456789",
        "tweets": [
            {
                "tweet_id": "101",
                "created_at": "Wed Oct 10 20:19:24 +0000 2018"
            },
            {
                "tweet_id": "102",
                "created_at": "2018-10-10T21:00:00Z"
            },
            {
                "tweet_id": "103",
                "created_at": "2018-10-10T22:00:00+01:00"
            }
        ]
    }
]

# Generate account-level predictions
predictions = predict_accounts(
    accounts=accounts,
    model=model,
    device=device
)

# Display predictions
for prediction in predictions:
    print(prediction)
```

### 10.1 Expected Output Structure

```python
{
    "user_id": "123456789",
    "bot_probability": 0.93,
    "predicted_label": 1
}
```

The probability shown above is illustrative. Actual results depend on the supplied tweets and the trained model.

### 10.2 Inference Configuration

The `predict_accounts()` function also supports optional parameters:

| Parameter     | Default       | Description                                                                                                    |
| ------------- | ------------- | -------------------------------------------------------------------------------------------------------------- |
| `model`       | `None`        | Preloaded model. If omitted, the model is loaded automatically.                                                |
| `device`      | `None`        | Inference device. Automatically selected when loading the model, or inferred from a supplied model if omitted. |
| `scaler_path` | `SCALER_PATH` | Path to the saved temporal scaler.                                                                             |
| `batch_size`  | `16`          | Number of accounts processed per inference batch.                                                              |
| `threshold`   | `0.5`         | Classification threshold for bot prediction.                                                                   |

For normal inference, the default configuration can be used.

## 11. Limitations and Responsible Interpretation

Agent 2 is a temporal bot-detection component trained and evaluated using the Cresci-2017 dataset. Its predictions should be interpreted within the limitations of this dataset and model.

### 11.1 Dataset Limitations

- Cresci-2017 represents a particular collection of Twitter accounts and may not reflect the behavior of all modern automated accounts.
- Dataset-specific characteristics and collection biases may influence model performance.
- The reported evaluation results do not guarantee equivalent performance on unseen platforms or different datasets.

### 11.2 Model Limitations

- Agent 2 analyzes temporal activity patterns only. It does not analyze tweet semantics or determine whether content is true or false.
- The model requires at least three tweets per account, consistent with the minimum sequence length represented in the training dataset.
- Its predictions depend on the availability and accuracy of tweet creation timestamps.
- The model is designed to analyze a collected account history, not to provide a real-time prediction after every individual tweet.

### 11.3 Responsible Interpretation

A high bot probability does not prove that an account is malicious, spreads misinformation, or is operated by an automated system.

Similarly, a low bot probability does not guarantee that an account is genuine or trustworthy.

Agent 2 should therefore be treated as one source of behavioral evidence within the multi-agent detection system, rather than as an independent authority for final moderation decisions.
