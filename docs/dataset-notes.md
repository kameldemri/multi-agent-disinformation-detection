# Dataset notes (project-wide)

This is the index of the raw datasets used by the project and the agreed handling rules.
Per-dataset EDA lives under [`docs/datasets/`](datasets/).

## 1. Raw dataset inventory

| Dataset | Raw location | Task / label source | Status | Details |
|---|---|---|---|---|
| Cresci-2017 | `data/raw/cresci_2017/` | Folder-derived: `genuine_accounts` = 0, `social_spambots` = 1 | EDA complete; not used for headline fake-news results | [Dataset EDA](datasets/cresci-2017.md) · [Agent data views](datasets/cresci-2017-agent-data.md) |
| Fake News Detection | `data/raw/fake_news/` | File-derived: fake / true articles | EDA complete; main content dataset | [Dataset EDA](datasets/fake-news.md) |
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
  claim-veracity label, so it is a supplementary bot-behaviour benchmark, not a fake-news
  dataset. Full numbers, limitations and leakage table: [Cresci-2017 EDA](datasets/cresci-2017.md).
- Notebooks (all outputs are displayed inline; no result files are written):
  [overview](../analysis/eda_cresci_overview.ipynb),
  [text](../agents/agent1_naive_bayes/analysis/eda_cresci_text.ipynb),
  [timing/user](../agents/agent2_rnn/analysis/eda_cresci_timing_user.ipynb).

## 3. Which agent uses which data

This is the only division done in this phase, and it is a **column/view split, not a row split**:

- Agent 1 (Naive Bayes): one tweet → `text` only, labelled spam (1) or not spam (0) by its user.
- Agent 2 (RNN): one ordered tweet sequence per user → `created_at` ordering, gaps, hour/day,
  tweets-per-user, labelled by the account.
- Agent 3: combined user-level view; no separate EDA notebook yet.
- Both agents read the same raw CSVs and the same shared checks (integrity, balance, leakage,
  `test_set_*` audit). See [Cresci-2017 agent data views](datasets/cresci-2017-agent-data.md).
- `test_set_1` / `test_set_2` are only audited as leakage columns.

## 4. Handoff boundary

This phase delivers the EDA, notebooks and column/view guidance. The teammates doing the modelling
own the train/validation/test split and training pipeline. For Cresci-2017 they must keep each
user's data together, exclude the leakage columns, handle the 2,391 genuine users with zero tweets,
and account for the repeated spambot text.

## 5. Reading order for teammates

1. This file — project-level map.
2. [Full Cresci-2017 EDA](datasets/cresci-2017.md) — all verified numbers, limitations, leakage table.
3. [Agent data views](datasets/cresci-2017-agent-data.md) — what agent 1 and agent 2 actually use.
4. [Overview notebook](../analysis/eda_cresci_overview.ipynb) — start at **Key takeaways**, then sections 1–9.
5. The focused notebook for your agent: [text](../agents/agent1_naive_bayes/analysis/eda_cresci_text.ipynb)
   or [timing/user](../agents/agent2_rnn/analysis/eda_cresci_timing_user.ipynb).

## 6. Open items

- Dataset license/version for Cresci-2017: TODO: confirm from the Kaggle page.
- Twitter Bot Detection dataset source, labels and EDA: TODO: confirm.

Date: 2026-11-22.
