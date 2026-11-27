# Dataset: Fake News Detection (Agent 1)

## 1. Source
- Kaggle: "Fake News Detection" by Bhavik Jikadara, https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection
- License: CC BY 4.0
- Version: 1 (116.37 MB). Downloaded on 2026-10-31.
- Original files: `fake.csv`, `true.csv` (unchanged, stored in `data/raw/fake_news/`, not committed).

## 2. Dimensions and columns
- `fake.csv`: 23,481 rows, 4 columns. `true.csv`: 21,417 rows, 4 columns.
- Columns: `title` (headline), `text` (article body), `subject` (category), `date` (publication date).

## 3. Labels and class distribution
- No label column: the class comes from the file (fake = 1, true = 0).
- Distribution: 52.3% fake / 47.7% true (balanced enough).

## 4. Missing values and duplicates
- Missing values: none in either file (title, text, subject and date are all complete in `fake.csv` and `true.csv`).
- Exact duplicate rows: 3 (fake), 206 (true). Duplicate titles: 5,578 (fake), 591 (true).
- Empty texts: 630 (fake), 1 (true). These are empty strings or only spaces, not NaN, so `isna()` does not detect them. 

## 5. Important EDA findings
- Text length (words): fake median = 363, mean = 423.2; true median = 359, mean = 385.6.
- Empty or very short texts: 630 empty (fake), 1 (true); 866 fake and 1 true text have fewer than 10 words.
- Most frequent words: fake: `trump`, `said`, `president`, `people`, `just`, `clinton`, `obama`, `like`, `donald`, `hillary`; true: `said`, `trump`, `reuters`, `president`, `state`, `government`, `states`, `house`, `republican`, `united`. 
- The word "Reuters" appears in 1.3% of fake texts and 99.8% of true texts.
- `subject` values: fake = News (9,050), politics (6,841), left-news (4,459), Government News (1,570), US_News (783), Middle-east (778); true = politicsNews (11,272), worldnews (10,145). 
- `date` range: fake from 2015-03-31 to 2018-02-19 (10 invalid dates); true from 2016-01-13 to 2017-12-31 (0 invalid). 
- `subject` has no overlap between fake and true, and `date` ranges differ: both are unreliable features. 

## 6. Data quality issues
- Duplicates and empty texts.
- Source signature ("CITY (Reuters) -") that reveals the class.
- `subject` and `date` are unreliable features.

## 7. Applied Agent 1 preprocessing
- Ignore `subject` and `date`; they are not model features.
- Remove texts shorter than 10 words and exact/near-exact text duplicates.
- Remove texts that appear in both classes with conflicting labels.
- Normalize the Reuters signature and apostrophes before vectorization.
- Use `title + text` as the model input.
- Vectorize with word TF-IDF or word counts; the implemented pipeline searches over both.
- Group by normalized title so near-duplicate articles stay in the same split.
- Use a grouped stratified train / validation / test split; fit the vectorizer on training data only.
- Calibrate the final Naive Bayes probabilities before passing scores to Agent 3.

## 8. Suitability and limitations
- Suitable for a first Naive Bayes baseline (text column, two balanced classes).
- Limits: news articles, not short social-media posts; the model may learn the source or style instead of the truth.

## 9. Suitability and Decision

### Why the Fake News dataset is used for Agent 1

Agent 1 is the content-analysis agent. The team decided to treat its task as fake-vs-true article classification, so the file-derived `fake = 1` / `true = 0` labels are the training targets. The implemented pipeline is a TF-IDF/count-vectorizer + Naive Bayes classifier with calibrated probabilities.

### Why Cresci-2017 is not used for Agent 1

Cresci-2017 has account-level genuine/spambot labels, not claim-veracity labels. It is used by Agent 2 for temporal bot detection, not by Agent 1. The text-focused Cresci notebook remains exploratory material only.

### Why the Twitter Bot Detection (Goyal) dataset is not used for Agent 1

Its `Bot Label` describes accounts, not articles, and each account appears only once with a single tweet. It provides no fake-vs-true article labels and no per-account article history.

### Decision

- Agent 1 uses `data/raw/fake_news/fake.csv` and `data/raw/fake_news/true.csv`.
- Agent 1 does **not** use Cresci-2017 or the Goyal Twitter Bot Detection dataset.
- Agent 2 uses Cresci-2017; see [Cresci-2017 agent data](cresci-2017-agent-data.md).
- Limitations still apply: source bias, long news articles, duplicate titles and unreliable `subject` / `date` columns.

