# Cresci-2017: which data each agent uses (EDA handoff)

This is the practical companion to the [full Cresci-2017 EDA](cresci-2017.md). It documents the
only division done in this phase: **which view of the same data each agent needs**. We do not
create train/validation/test files here; the teammates doing the modelling own that step.

## 1. Same raw data, different views

Both agents read the same four raw files:

- `data/raw/cresci_2017/genuine_accounts/users.csv` → label 0, 3,474 users.
- `data/raw/cresci_2017/genuine_accounts/tweets.csv` → 2,839,362 tweets.
- `data/raw/cresci_2017/social_spambots/users.csv` → label 1, 991 users.
- `data/raw/cresci_2017/social_spambots/tweets.csv` → 1,610,034 tweets.

Join `tweets.user_id` to `users.id`. A tweet inherits its user's account label. No rows are given
exclusively to one agent: both notebooks stream all four CSVs in 100,000-row chunks and each one
focuses on the columns relevant to that agent.

## 2. Agent views at a glance

| Agent | Notebook | Unit | Columns / signals used | Label |
|---|---|---|---|---|
| Agent 1 — Naive Bayes | `agents/agent1_naive_bayes/analysis/eda_cresci_text.ipynb` | one tweet | `text` only; char/word length, duplicate text, hashtags/mentions/URLs, script, word/hashtag frequencies | account label (0/1) inherited from the tweet's user |
| Agent 2 — RNN | `agents/agent2_rnn/analysis/eda_cresci_timing_user.ipynb` | one ordered tweet sequence per user | `user_id` for grouping, `created_at` for ordering; tweets/user, inter-tweet gaps, hour/day, user metadata | account label (0/1) |
| Agent 3 — combined | no separate EDA notebook yet | one user | text evidence + timing/user evidence together | account label (0/1) |
| Shared checks | overview + both agent notebooks | dataset-level | integrity, join, class balance, leakage audit, `test_set_*` audit | — |

## 3. Agent 1: text / spam-or-not-spam view

**What the model sees:** one row = one tweet with its text, labelled by the account that posted it
(genuine → 0, spambot → 1). Use `text` as the input; derive token/character features from it.
Do not feed IDs, source, time, metadata or `test_set_*` columns.

Evidence from the EDA:

- Median length: 64 chars genuine vs 87 spambot; mean words 11.14 vs 14.76.
- Retweets using the non-sentinel definition: 738,089 genuine (25.99%) vs 37,522 spambot (2.33%).
  `retweeted_status_id=0` is a sentinel, not a real retweet.
- Within-class duplicate text: 4.37% genuine vs 90.30% spambot. Cross-class exact text overlap is
  tiny: 0.04% vs 0.02%. Spambot text is highly repetitive, with one text repeated up to 5,145 times.
- Mean hashtags/mentions/URLs: genuine 0.2477 / 0.7714 / 0.1507; spambot 0.0954 / 0.0421 / 0.0609.
- Latin script: 98.59% genuine vs 99.98% spambot. Top words differ sharply: genuine `rt`, `the`,
  `to`, `you`, `and`; spambot `che`, `di`, `non`, `la`, `il`.
- Missing text: 12,572 genuine tweets and 90 spambot tweets.

Things to handle, not model features:

- The account label is reused across many tweets; keep a user's tweets together in any later split.
- Duplicate spambot text can dominate frequency features; deduplicate or cap repetitions.
- `source` is a strong text-adjacent leak (TweetAdder v4 is the top spambot source): drop it.

Preprocessing recipe for the teammate:

1. Stream each `tweets.csv` in chunks; keep `text` and `user_id` only.
2. Set `y=0` for tweets from `genuine_accounts/`, `y=1` for `social_spambots/`. Drop the
   12,572 genuine + 90 spambot rows with missing text.
3. Keep `user_id` only as a grouping key for the later split. It is not an input feature.
4. Normalize text (lowercase, strip markup/URLs/mentions if that is the chosen convention), then
   build token/character features with `CountVectorizer` or `TfidfVectorizer`. Fit the vectorizer
   on training data only.
5. Deduplicate or cap repeated text: spambots have 189,997 distinct texts across 1,609,944
   text-bearing tweets, and one text appears 5,145 times.
6. Train Naive Bayes on `X = text features`, `y = account label`. Retweet status is metadata, not
   text; Agent 1 is specified as text-only, so leave `retweeted_status_id` out unless the team
   deliberately changes the scope.

## 4. Agent 2: temporal / RNN view

**What the model sees:** one sample = one user, represented as an ordered sequence of that user's
tweets sorted by `created_at`, with derived gaps/hour/day/tweet-count features. Label is the
account label.

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
- User numeric medians: `statuses_count` 6,609 vs 138; `favourites_count` 1,286 vs 0;
  `listed_count` 2 vs 0; `followers_count` 341 vs 234; `friends_count` 319 vs 313.

Things to handle, not model features:

- Use `created_at` for ordering. `timestamp` is shifted +3600/+7200 s and is redundant.
- Never use `crawled_at`, `updated`, `test_set_1` or `test_set_2`; they are leakage columns.
- Drop `source`, `lang`, `geo_enabled`, `verified`, `time_zone`, `listed_count` and the other
  MEDIUM/HIGH columns in section 7 of the main EDA. They describe the collection artifact, not
  posting behaviour.
- Users with zero tweets are a coverage limitation of this dataset, not missing values to impute
  into a sequence.

Preprocessing recipe for the teammate:

1. Stream the tweet files; keep `user_id`, `id` (tie-break only) and `created_at`. Parse
   `created_at` with the Twitter API format; drop the 1 genuine row with a null time.
2. Group by `user_id`, sort ascending by `created_at`, and break exact timestamp ties with `id`
   (tied timestamps are common: 14,844 genuine / 6,629 spambot tweets).
3. For each user, compute `gap_seconds` between consecutive tweets and derive hour/day-of-week
   from `created_at` UTC. Gaps are extremely skewed (max 113,836,203 s genuine / 67,582,725 s
   spambot), so log-transform or clip them.
4. Build one sample per user with ≥1 tweet. Features are a padded/truncated sequence of
   `(gap, hour, day_of_week)` steps. Optionally append safe profile numerics
   (`statuses_count`, `followers_count`, `friends_count`, `favourites_count`) as static features;
   keep `listed_count` and all MEDIUM/HIGH leakage columns out.
5. Cap the sequence length to a team-chosen N and pad shorter sequences. Full sequences are
   impractical: p95 is about 3,224 genuine / 3,213.5 spambot tweets per user.
6. Exclude the 2,391 genuine users with zero tweets from temporal training. They cannot produce a
   sequence and should not be imputed with fake events.
7. Fit scalers/encoders on training users only, and keep all tweets of a user in the same split.

## 5. Shared checks in every notebook

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

This phase delivers the EDA, the notebooks, the column/view guidance and the preprocessing
recipes above. The teammates doing the modelling choose the split sizes, seed, metrics, model
hyperparameters and exact text-normalization/truncation settings. The EDA constraints still apply:
keep each user's data together, exclude leakage columns, fit transformers on training data only,
and account for the zero-tweet genuine users and repeated spambot text.

Date: 2026-11-22.
