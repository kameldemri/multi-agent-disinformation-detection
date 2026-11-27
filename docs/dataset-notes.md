# Dataset notes (project-wide)

This is the index of the raw datasets used by the project and the agreed handling rules.
Per-dataset EDA lives under [`docs/datasets/`](datasets/).

## 1. Raw dataset inventory

| Dataset | Raw location | Task / label source | Status | Details |
|---|---|---|---|---|
| Cresci-2017 | `data/raw/cresci_2017/` | Folder-derived: `genuine_accounts` = 0, `social_spambots` = 1 | EDA complete; used by Agent 2 temporal model | [Dataset EDA](datasets/cresci-2017.md) · [Agent data views](datasets/cresci-2017-agent-data.md) |
| Fake News Detection | `data/raw/fake_news/` | File-derived: fake = 1, true = 0 | EDA complete; used by Agent 1 Naive Bayes | [Dataset EDA](datasets/fake-news.md) |
| Twitter Bot Detection | `data/raw/twitter_bot_detection/` | TODO: confirm | Not yet inspected | [Notes](datasets/twitter-bot.md) (empty) |

All raw CSVs are gitignored. Cresci-2017 can be downloaded from
https://www.kaggle.com/datasets/hashemalsalmi/cresci-2017 and must keep the folder names
`genuine_accounts/` and `social_spambots/` with their inner `tweets.csv` / `users.csv`.

## 2. Overall Cresci-2017 dataset

- Two class folders, each with one tweet file and one user file: 4,465 accounts and 4,449,396
  tweets in total.
- User balance: 3,474 genuine / 991 spambot (77.81% / 22.19%).
- Tweet balance: 2,839,362 genuine / 1,610,034 spambot (63.81% / 36.19%).
- Labels are account-level. The dataset contains one bot type only (social spambots) and no
  claim-veracity label, so it is a bot-behaviour benchmark for Agent 2, not a fake-news dataset.
  Full numbers, limitations and leakage table: [Cresci-2017 EDA](datasets/cresci-2017.md).
- Notebooks (all outputs are displayed inline; no result files are written):
  [overview](../analysis/eda_cresci_overview.ipynb),
  [text](../agents/agent1_naive_bayes/analysis/eda_cresci_text.ipynb),
  [timing/user](../agents/agent2_rnn/analysis/eda_cresci_timing_user.ipynb).

## 3. Which agent uses which data

The agents use different raw datasets; there is no shared row split between them.

- Agent 1 (Naive Bayes): one news article from `fake.csv` / `true.csv` → `title + text`, labelled
  fake (1) or true (0) by the source file. See [Fake News EDA](datasets/fake-news.md).
- Agent 2 (RNN): one ordered tweet sequence per Cresci-2017 user → `created_at` ordering, gaps,
  hour/day, tweets-per-user, labelled genuine (0) or spambot (1) by the account. See
  [Cresci-2017 agent data views](datasets/cresci-2017-agent-data.md).
- Agent 3: combined user-level view; no separate EDA notebook yet.
- The Cresci-2017 notebooks still document a text-only exploration, but that view is exploratory
  only and is not the Agent 1 training recipe.
- In Cresci-2017, `test_set_1` / `test_set_2` are audited as leakage columns only.

## 4. Handoff boundary

The EDA phase delivered dataset notes, notebooks and column/view guidance. The implemented modelling
pipelines own their own train/validation/test split and training:

- Agent 1 handles the Fake News dataset, groups near-duplicate articles by normalized title, and
  excludes `subject` and `date`.
- Agent 2 handles Cresci-2017, keeps each user's tweets together, excludes the leakage columns,
  handles the 2,391 genuine users with zero tweets, and scales temporal features using training data
  only.

## 5. Reading order for teammates

1. This file — project-level map.
2. Agent 1: [Fake News EDA](datasets/fake-news.md) and [`agents/agent1_naive_bayes/README.md`](../agents/agent1_naive_bayes/README.md).
3. Agent 2: [Full Cresci-2017 EDA](datasets/cresci-2017.md), [Agent data views](datasets/cresci-2017-agent-data.md) and [`agents/agent2_rnn/README.md`](../agents/agent2_rnn/README.md).
4. Cresci notebooks: [overview](../analysis/eda_cresci_overview.ipynb) and [timing/user](../agents/agent2_rnn/analysis/eda_cresci_timing_user.ipynb).
5. The text-focused Cresci notebook is archived exploration: [`eda_cresci_text.ipynb`](../agents/agent1_naive_bayes/analysis/eda_cresci_text.ipynb).

## 6. Open items

- Dataset license/version for Cresci-2017: TODO: confirm from the Kaggle page.
- Twitter Bot Detection dataset source, labels and EDA: TODO: confirm.

Date: 2026-11-22.
