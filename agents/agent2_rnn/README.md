# Agent 2 — temporal BiGRU bot detector

Agent 2 classifies one Cresci-2017 account as genuine (`0`) or spambot (`1`) from its ordered tweet
sequence. Raw data lives in `data/raw/cresci_2017/` and is gitignored.

## Files

- `src/preprocess.py` — seven preprocessing phases and the `--phase` CLI.
- `src/dataset.py` — `.npz` loading, validation, length-based batching and dynamic padding.
- `src/model.py` — bidirectional GRU.
- `src/baseline.py` — activity-volume baselines.
- `src/train.py` — training loop, early stopping and checkpointing.
- `src/predict.py` — inference API and CLI.
- `notebooks/Agent2_BiGRU_Training.ipynb` — Colab training/report notebook.

## Setup

PyTorch is required. Install from the repo root:

```bash
pip install -r requirements.txt
```

## Inference artifacts

Prediction does not need the raw tweet CSVs. Single-sequence scoring needs
`checkpoints/agent2/best_bigru.pt` and `data/processed/agent2/temporal_scaler.npz`.
`--split test` also needs `data/processed/agent2/account_splits.csv` and
`data/processed/agent2/test_scaled_sequences.npz`.

## Preprocess

From the repo root:

```bash
python -m agents.agent2_rnn.src.preprocess --phase all
```

Individual phases are `--phase 1` through `--phase 7`. Outputs are written under
`data/processed/agent2/` and are gitignored.

## Train

From the repo root:

```bash
python -m agents.agent2_rnn.src.train
```

Outputs:

- `checkpoints/agent2/best_bigru.pt`
- `checkpoints/agent2/training_history.csv`

Training uses train/validation splits. The saved notebook reports validation metrics; a final
test-set evaluation is still pending.

## Predict

From the repo root:

```bash
python -m agents.agent2_rnn.src.predict --split test --output results/agent2_test_scores.csv
```

`--split test` writes `user_id`, `score` and `label`. Raw sequences can also be scored with
`--sequences-json` or `--sequences-file`.

Public API in `src/predict.py`:

- `load_model(checkpoint_path, scaler_path, device)` — load model and scaler.
- `prepare_sequence(sequence, mean, std)` — transform/scaling helper.
- `score_account(sequence, ...)` — score one account.
- `score_accounts(sequences, ...)` — batch scoring.
- `score_split(split, ...)` — aligned scores and labels for a processed split.
