# Cresci-2017: Agent 2 temporal data handoff

This is the practical companion to the [full Cresci-2017 EDA](cresci-2017.md). It documents the
view used by Agent 2 and the constraints inherited from EDA. This project phase does not create
model splits; the Agent 2 pipeline owns the train/validation/test files and training. Agent 1 uses
the [Fake News dataset](fake-news.md), not Cresci-2017.

## 1. Raw data used by Agent 2

- `data/raw/cresci_2017/genuine_accounts/users.csv` → label 0, 3,474 users.
- `data/raw/cresci_2017/genuine_accounts/tweets.csv` → 2,839,362 tweets.
- `data/raw/cresci_2017/social_spambots/users.csv` → label 1, 991 users.
- `data/raw/cresci_2017/social_spambots/tweets.csv` → 1,610,034 tweets.

Join `tweets.user_id` to `users.id`. A tweet inherits its user's account label. The timing/user
notebook streams all four CSVs in 100,000-row chunks and focuses on ordering, gaps and account
metadata.

## 2. Agent views at a glance

| Agent | Dataset | Unit | Columns / signals used | Label |
|---|---|---|---|---|
| Agent 1 — Naive Bayes | `data/raw/fake_news/` | one news article | `title + text`; `subject` and `date` are excluded | fake = 1, true = 0 |
| Agent 2 — RNN | `data/raw/cresci_2017/` | one ordered tweet sequence per user | `user_id` for grouping, `created_at` for ordering; inter-tweet gaps, hour and day-of-week | account label (0 = genuine, 1 = spambot) |
| Agent 3 — combined | both agent outputs | one account | Agent 2 score + Agent 1 article evidence when available | risk decision |
| Shared checks | Cresci-2017 notebooks | dataset-level | integrity, join, class balance, leakage audit, `test_set_*` audit | — |

## 3. Archived text-only Cresci exploration

The text-focused notebook
[`eda_cresci_text.ipynb`](../../agents/agent1_naive_bayes/analysis/eda_cresci_text.ipynb)
is kept as exploratory material. It is not the Agent 1 training recipe.

Useful findings from that exploration:

- Median tweet length: 64 chars genuine vs 87 spambot; mean words 11.14 vs 14.76.
- Within-class duplicate text: 4.37% genuine vs 90.30% spambot. Cross-class exact text overlap is
  tiny: 0.04% vs 0.02%. One spambot text is repeated up to 5,145 times.
- Mean hashtags/mentions/URLs: genuine 0.2477 / 0.7714 / 0.1507; spambot 0.0954 / 0.0421 / 0.0609.
- Missing text: 12,572 genuine tweets and 90 spambot tweets.
- `source` is a strong leak and must not be used as a feature.

Do not train Agent 1 on this view. It has account-level labels, not claim-veracity labels.

## 4. Agent 2: temporal / RNN view

**What the model sees:** one sample = one user, represented as an ordered sequence of that user's
tweets sorted by `created_at`, with derived gaps/hour/day features. Label is the account label.

Evidence from the EDA:

- Usable users with at least one tweet: 1,083 genuine + 991 spambot = 2,074.
  2,391 genuine users have zero tweets and cannot form a temporal sequence.
- Median tweets/user: 0 genuine (all users) vs 1,387 spambot; among users with tweets, 3,187 vs
  1,387. p95 is 3,224 vs 3,213.5.
- Inter-tweet gaps: median 423 s genuine vs 4,166 s spambot; p25 72 vs 824 s; p75 5,588 vs
  14,381 s. Gaps under 60 s: 22.23% vs 3.87%.
- Tied posting timestamps: 557 genuine users / 14,844 tweets vs 297 spambot users / 6,629 tweets;
  max tied tweets for one user 2,333 vs 2,935.
- UTC hour distribution: genuine is flatter (2.51%–5.58% per hour); spambots concentrate from
  05:00–21:00 (peak 17:00 = 6.28%) and are almost absent at 23:00–03:00 (0.21%–0.71%).

Things to handle, not model features:

- Use `created_at` for ordering. `timestamp` is shifted +3600/+7200 s and is redundant.
- Never use `crawled_at`, `updated`, `test_set_1` or `test_set_2`; they are leakage columns.
- Drop `source`, `lang`, `geo_enabled`, `verified`, `time_zone`, `listed_count` and the other
  MEDIUM/HIGH columns in section 7 of the main EDA.
- Users with zero tweets are a coverage limitation, not missing values to impute into a sequence.

Implemented preprocessing in `agents/agent2_rnn/src/preprocess.py`:

1. Filter tweet rows to known users with a valid `created_at`.
2. Create account-level train/validation/test assignments; the current run uses 1,451 / 311 / 312
   accounts.
3. Store tweets in SQLite, import both classes and verify labels/splits.
4. Export temporal features: `gap_seconds`, `hour`, `day_of_week`.
5. Build one variable-length sequence per account and retain 4,449,395 tweets.
6. Transform each timestep to `[log_gap, hour_sin, hour_cos, day_sin, day_cos]`.
7. Fit the scaler on training accounts only, then apply it to all splits.

The implemented model uses dynamic padding and length-sorted batches instead of truncating
sequences. Static profile numerics are not used. Scaler parameters are stored in
`data/processed/agent2/temporal_scaler.npz`.

## 5. Shared checks in every Cresci notebook

- **Section 1, structure/integrity:** shapes, dtypes, missing values, exact duplicates (0).
- **Section 2, join/balance:** 3,474/991 users, 2,839,362/1,610,034 tweets, 1 null/missing genuine
  user id, 2,391 genuine users with zero tweets.
- **Section 8, leakage audit:** HIGH = `test_set_1`, `test_set_2`, `crawled_at`, `updated`,
  `source`, `lang`, `geo_enabled`; MEDIUM = `verified`, `time_zone`, `location`, `listed_count`,
  `utc_offset`, `protected`, `profile_use_background_image`, `profile_sidebar_border_color`;
  LOW = `url`.
- **Section 9, `test_set_*` audit:** all 991 spambots have `test_set_1=1`; `test_set_2` exists only
  in the genuine user file. These columns are audited as leakage evidence only — we do not build
  any split from them.

## 6. Handoff boundary

The EDA phase delivered dataset notes, notebooks and constraints. The Agent 2 pipeline owns the
split sizes, seed, metrics, hyperparameters and exact sequence/feature settings. The EDA constraints
still apply: keep each user's tweets together, exclude leakage columns, fit transformers on training
data only, and account for the zero-tweet genuine users and repeated spambot text.

Date: 2026-11-22.
