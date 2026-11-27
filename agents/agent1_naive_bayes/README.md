# Agent 1 — Naive Bayes fake-news classifier

Agent 1 classifies one news article as fake (`1`) or true (`0`) using
`data/raw/fake_news/fake.csv` and `data/raw/fake_news/true.csv`. Raw CSVs are gitignored.

## Files

- `src/preprocess.py` — loads raw CSVs, cleans text, removes short/duplicate/ambiguous rows,
  builds grouped stratified train/validation/test indices.
- `src/train.py` — grid search, final fit, probability calibration and evaluation.
- `src/predict.py` — inference API and CLI.

## Public API

From `src/predict.py`:

- `load_model(path)` — load a saved bundle.
- `score_text(text, title="")` — calibrated fake probability in `[0, 1]`.
- `score_texts(texts, titles=None)` — batch scoring.
- `explain_text(text, title="", n=8)` — decisive words for fake and true.

## Inference artifacts

Prediction does not need the raw CSVs. It needs the trained bundle:
`models/nb_fake_news.joblib`. The bundle contains the vectorizer, Naive Bayes model, calibrator and
metadata.

## Train

From `agents/agent1_naive_bayes/`:

```bash
python -m src.train --quick
```

Outputs:

- `models/nb_fake_news.joblib`
- `results/` (metrics, plots and `summary.json`)

Use `--experiments` for the larger report grid. Use `--raw-dir`, `--model-path` and
`--results-dir` to override defaults.

## Predict

From `agents/agent1_naive_bayes/`:

```bash
python -m src.predict --explain "article text"
python -m src.predict --file article.txt --title "headline"
```

The score is the probability that the article is fake. Agent 3 should consume this score, not
the raw `subject` or `date` columns.
