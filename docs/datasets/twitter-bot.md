# Dataset: Twitter Bot Detection (Agent 2)

## 1. Source

- Dataset: Twitter Bot Detection Dataset by Aditya Goyal.
- Local file: `bot_detection_data.csv`.
- The raw dataset is stored locally and is not committed to the repository.

## 2. Dimensions and columns

- Total records: 50,000.
- Total columns: 11.
- Columns:
  - `User ID`: identifier of the Twitter account.
  - `Username`: account username.
  - `Tweet`: text content of the tweet.
  - `Retweet Count`: number of retweets.
  - `Mention Count`: number of mentions.
  - `Follower Count`: account follower count.
  - `Verified`: account verification status.
  - `Bot Label`: account classification (bot or human).
  - `Location`: account location.
  - `Created At`: tweet publication timestamp.
  - `Hashtags`: hashtags included in the tweet.

## 3. Labels and class distribution

- The dataset contains a `Bot Label` column that identifies whether an account is classified as a bot or a human.
- The label describes the account, not the individual tweet.
- The dataset contains 50,000 records, but the class distribution should be interpreted at the account level.

## 4. Missing values and duplicates

- The EDA confirmed that the dataset contains no missing timestamps in `Created At`.
- There are 50,000 unique user IDs across 50,000 records.
- Each user appears exactly once in the dataset.
- No account has multiple tweet records in this dataset.

## 5. Important EDA findings

### User distribution

- Number of records: 50,000.
- Number of unique users: 50,000.
- Each account is represented by exactly one tweet.
- No repeated user histories are available.

### Temporal analysis

- The `Created At` column contains valid timestamps for all records.
- The observed date range is from **2020-01-01 to 2023-05-31**.
- The timestamps allow the records to be ordered chronologically and compared at the platform level.

### Main structural finding

The most important finding is that the dataset contains only one tweet per user. Consequently, it does not provide the sequence of publications needed to study the temporal behavior of individual accounts.

## 6. Data quality and structural limitations

- The dataset contains one record per user, so it cannot directly support the analysis of individual account publication histories.
- It is not possible to calculate reliable per-account posting frequency, inter-post intervals, or individual temporal activity patterns from this dataset.
- Chronologically sorting all records produces a sequence of tweets from different users. This is a platform-level sequence, not an individual account's publication history.
- The `Bot Label` is an account-level label. It does not identify coordinated campaigns or provide campaign-level ground truth.

## 7. Possible uses and alternatives

Despite its structural limitation for individual temporal analysis, the dataset can still support:

- Exploratory analysis of bot and human account characteristics.
- Comparison of tweet engagement indicators, such as retweets and mentions.
- Exploration of publication timestamps at the platform level.
- Preliminary investigation of similarities between posts from different accounts.

However, these analyses should not be presented as proof of coordinated behavior or as a substitute for individual account histories.

## 8. Suitability and limitations

- Suitable for exploratory analysis of account-level bot and human characteristics.
- Contains tweet text, timestamps, engagement indicators, and account labels.
- Not suitable for training or validating an RNN that requires sequences of publications from individual accounts.
- Does not provide campaign labels for supervised campaign-level coordination detection.

## 9. Suitability and Decision

### Why the dataset is limited for Agent 2's temporal analysis

Agent 2 is responsible for analyzing temporal publication behavior using an RNN. This requires multiple timestamped publications associated with the same account so that the model can learn patterns such as posting frequency and intervals between publications.

However, this dataset contains 50,000 records belonging to 50,000 unique users. Each user appears only once. Therefore, individual account publication sequences cannot be constructed.

### Preliminary modeling observation

A preliminary experiment aggregated the records by day and used an RNN to forecast platform-level activity indicators.

Although the model performed better than a persistence baseline on MAE and RMSE, its predictions showed substantially lower variance than the actual values and did not adequately follow daily fluctuations.

This experiment is exploratory and does not establish that the model can detect bots or coordinated campaigns.

### Decision

- The dataset is retained as preliminary EDA evidence.
- It is not considered sufficient for the final individual-account temporal analysis required by Agent 2.
- A replacement dataset should contain multiple timestamped tweets per account, stable user identifiers, and preferably bot/human labels.
- Campaign or coordination labels would also be valuable if campaign-level detection is part of the final evaluation.
