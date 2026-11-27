# Dataset: Cresci-2017 (genuine accounts vs social spambots)

Full EDA handoff for the team. Everything here comes from the executed notebook outputs; the
notebooks are the evidence and this file is the written report.

Notebooks:
- [Overview EDA](../../analysis/eda_cresci_overview.ipynb) — all sections, whole dataset.
- [Text-focus EDA](../../agents/agent1_naive_bayes/analysis/eda_cresci_text.ipynb) — sections 1, 2, 6, 8, 9.
- [Timing/user-focus EDA](../../agents/agent2_rnn/analysis/eda_cresci_timing_user.ipynb) — sections 1, 2, 3, 4, 5, 7, 8, 9.
- [Agent 2 data handoff](cresci-2017-agent-data.md) — practical temporal-view recipe for Agent 2.

Source: Kaggle, https://www.kaggle.com/datasets/hashemalsalmi/cresci-2017.
Raw data is gitignored under `data/raw/cresci_2017/`.

## 1. Data model: what you actually have

Two class folders, each with one user file and one tweet file:

| Folder | Label | `users.csv` | `tweets.csv` | File sizes measured by EDA |
|---|---:|---:|---:|---|
| `genuine_accounts/` | 0 (not spam / genuine) | 3,474 rows × 42 columns | 2,839,362 rows × 25 columns | 2.3 MB / 964.6 MB |
| `social_spambots/` | 1 (spam / spambot) | 991 rows × 41 columns | 1,610,034 rows × 25 columns | 0.6 MB / 534.6 MB |
| **Total** | — | **4,465 users** | **4,449,396 tweets** | — |

- Labels come from the **folder**, not from a column. Genuine = 0, spambot = 1.
- The label is **account-level**. A tweet inherits the label of its user.
- Join `tweets.user_id` to `users.id`.
- Genuine `users.csv` has one extra column: `test_set_2` (42 vs 41 columns).
- Never open the tweet CSVs in Excel or load them naively into memory; the notebooks stream
  them in 100,000-row chunks.

## 2. Class balance and integrity

| Item | Genuine | Spambot | Total |
|---|---:|---:|---:|
| Users | 3,474 | 991 | 4,465 |
| User share | 77.81% | 22.19% | 100% |
| Tweets | 2,839,362 | 1,610,034 | 4,449,396 |
| Tweet share | 63.81% | 36.19% | 100% |
| Users with zero tweets | 2,391 | 0 | 2,391 |
| Users with ≥1 tweet | 1,083 | 991 | 2,074 |
| Tweets with null `user_id` | 1 | 0 | 1 |
| Tweets whose user id is missing from `users.csv` | 1 | 0 | 1 |
| Exact duplicate rows (tweets / users) | 0 / 0 | 0 / 0 | 0 |

- Imbalance: 3.51 genuine users per spambot user; 1.76 genuine tweets per spambot tweet.
- Missing text: 12,572 genuine tweets (0.44%) and 90 spambot tweets (0.006%).
- `truncated`, `favorited` and `retweeted` are 100% missing in both tweet files.
- `in_reply_to_status_id` and `in_reply_to_user_id` have one missing genuine row each.
- No parse failures, epoch-0 timestamps, or future-dated values were found.

## 3. Time columns: usable and unusable

| Column | What it is | Verdict |
|---|---|---|
| `created_at` (tweets) | Twitter API time with `+0000`; parsed cleanly except 1 genuine null | **Use for temporal analysis** |
| `created_at` (users) | account creation time; ranges genuine 2007-01-22 → 2015-04-20, spambot 2009-03-17 → 2014-05-26 | Descriptive; overlapping range |
| `timestamp` | naive local time; never equals parsed `created_at` | Redundant/shifted: +3600 s for 49.26% genuine / 35.42% spambot; +7200 s for the rest |
| `crawled_at` | collection time | **Leakage: drop** (100% class-exclusive values) |
| `updated` | last collection update | **Leakage: drop** (100% class-exclusive values) |

Tweet `created_at` ranges: genuine 2007-03-10 22:25:59+00:00 → 2015-05-01 14:04:57+00:00;
spambot 2009-03-17 08:57:10+00:00 → 2014-11-13 22:11:22+00:00.
`crawled_at` and `updated` are almost always equal (99.9988% genuine / 99.9996% spambot rows);
they describe when the dataset was crawled, not when users posted.

## 4. Posting-volume and temporal signals

| Signal | Genuine | Spambot |
|---|---:|---:|
| Median tweets per user (all users) | 0 | 1,387 |
| p95 tweets per user | 3,224 | 3,213.5 |
| Median tweets per user (users with ≥1 tweet) | 3,187 | 1,387 |
| Users with ≥100 tweets | 1,058 | 991 |
| Gaps analysed (users with ≥2 posts) | 2,838,278 | 1,609,043 |
| Median inter-tweet gap | 423 s | 4,166 s |
| p25 / p75 gap | 72 s / 5,588 s | 824 s / 14,381 s |
| p95 gap | 75,051 s | 54,615 s |
| Gaps under 60 s | 22.23% | 3.87% |
| Zero-length gaps | 0.33% | 0.27% |
| Users with tied posting timestamps | 557 | 297 |
| Tweets inside tied timestamp groups | 14,844 | 6,629 |
| Max tied tweets for one user | 2,333 | 2,935 |

Hour-of-day (UTC) is also different: genuine hours range from 2.51% (10:00) to 5.58% (03:00)
share of tweets, while spambots concentrate in 05:00–21:00 (peak 17:00 at 6.28%) and are almost
absent late night (23:00 = 0.71%, 00:00 = 0.26%). Full per-hour tables are in section 5 of the
timing/user notebook.

## 5. Text signals

| Signal | Genuine | Spambot |
|---|---:|---:|
| Tweets with text | 2,826,790 | 1,609,944 |
| Median char length | 64 | 87 |
| Mean char length | 71.44 | 87.58 |
| p25 / p75 char length | 38 / 104 | 62 / 116 |
| Median words | 10 | 15 |
| Mean words | 11.14 | 14.76 |
| Retweets (`retweeted_status_id` non-null and ≠ 0) | 738,089 (25.99%) | 37,522 (2.33%) |
| Within-class duplicate text | 123,573 (4.37%) | 1,453,732 (90.30%) |
| Cross-class exact text overlap | 1,206 (0.04%) | 285 (0.02%) |
| Distinct texts | 2,740,774 | 189,997 |
| Max copies of one text | 1,025 | 5,145 |
| Mean `num_hashtags` / `num_mentions` / `num_urls` | 0.2477 / 0.7714 / 0.1507 | 0.0954 / 0.0421 / 0.0609 |
| Distinct hashtags / mentions | 162,878 / 324,968 | 9,482 / 10,133 |
| Latin script share | 98.59% | 99.98% |

Top words: genuine `rt`, `the`, `to`, `you`, `and`; spambot `che`, `di`, `non`, `la`, `il`
(Italian). Top hashtags: genuine `kca`, `vote5sos`, `vote1duk`; spambot `sapevatelo`,
`proverbi`, `news`.

`retweeted_status_id` uses `0` as a sentinel for non-retweets; raw non-null counts are therefore
misleading (99.999965% genuine / 100% spambot). Use the non-null-and-not-0 numbers above.

## 6. User metadata signals

| Column | Genuine median | Spambot median | Note |
|---|---:|---:|---|
| `statuses_count` | 6,609 | 138 | Genuine accounts claim far more past statuses |
| `followers_count` | 341 | 234 | Overlap exists |
| `friends_count` | 319 | 313 | Almost identical median |
| `favourites_count` | 1,286 | 0 | Strong separation |
| `listed_count` | 2 | 0 | See leakage caveat below |

`statuses_count` divided by the number of tweets actually present has median 2.60 for genuine
but 0.10 for spambots; only 9.44% of genuine and 1.01% of spambot users are within ±10% of 1.
Categorical differences: `verified` appears in 11 genuine users (0.32%) and no spambots;
`geo_enabled=1` in 2,155 genuine (62.03%) vs 18 spambots (1.82%); `lang=it` in 900 spambots
(90.82%) and no genuine; `time_zone=Athens` in 831 spambots (83.85%) and no genuine.
These are informative descriptively but are leakage-prone as features (section 7).

## 7. Leakage risks

Verdicts come from the leakage-audit cell of the notebooks.

| Column | Source file | Verdict | Evidence from the EDA |
|---|---|---|---|
| `test_set_1` | users.csv | HIGH | Genuine: 2,474 (71.2%) at 0; all 991 spambots at 1 |
| `test_set_2` | users.csv | HIGH (structural) | Column absent from spambot `users.csv` (42 vs 41 columns) |
| `crawled_at` | users.csv / tweets.csv | HIGH | Values are 100% class-exclusive in both files (3,673 / 12,433 distinct) |
| `updated` | users.csv / tweets.csv | HIGH | Values are 100% class-exclusive in both files (1,482 / 12,547 distinct) |
| `source` | tweets.csv | HIGH | 270 spambot-only values covering 92.7% of spambot rows; top spambot source TweetAdder v4 (2.95%) |
| `lang` | users.csv | HIGH | `it` is spambot-only, covering 90.8% of spambot rows |
| `geo_enabled` | users.csv | HIGH | Max class purity 0.9917 at `1` (n=2,173) |
| `verified` | users.csv | MEDIUM | 11 genuine users at `1`; 0 spambots |
| `time_zone` | users.csv | MEDIUM | 97 genuine-only values covering 56.8% of genuine rows; Athens covers 83.85% of spambots |
| `location`, `listed_count`, `utc_offset`, `protected`, `profile_use_background_image`, `profile_sidebar_border_color` | users.csv | MEDIUM | See the leakage-audit cell for per-column evidence |
| `url` | users.csv | LOW | Max class purity 0.6950 at `<MISSING>` (n=3,177) |

`test_set_1`, `test_set_2`, `crawled_at`, `updated`, `source`, `lang` and `geo_enabled` must not be
used as model features. The built-in `test_set_*` columns are audited only as evidence; we do not
build any train/validation/test split in this phase.

## 8. Suitability and decision

- Suitable as a **bot-detection benchmark for Agent 2**, which uses timing/user features.
- Not used by Agent 1: it has no claim-veracity label. Agent 1 uses the
  [Fake News dataset](fake-news.md) instead.
- One bot type only (`social_spambots`), so results will not generalise to other bot families.
- Account-level labels mean a tweet-level model would reuse the same label for thousands of tweets
  from one user; the EDA flags this explicitly.
- Class imbalance (3.51× users, 1.76× tweets) means a single accuracy number would be misleading.
- License/version: TODO: confirm from the Kaggle page.

## 9. How to use this EDA

1. Read this file for the whole picture and the verified numbers.
2. Read [Agent 2 data handoff](cresci-2017-agent-data.md) for the exact temporal view and implemented preprocessing recipe.
3. Open the overview notebook, start at the top **Key takeaways** cell, then follow sections 1 → 9.
4. Agent teammates open their own focused notebook; they contain the same shared checks plus only
   their sections.
5. The team members handling modelling own the train/validation/test split and training. The EDA
   only supplies constraints: keep a user's tweets together, exclude the leakage columns, handle
   the 2,391 genuine users with zero tweets, and be aware of duplicate spambot text.

## 10. Date

EDA run date: 2026-11-22 (recorded in the notebooks' run-metadata cell).
