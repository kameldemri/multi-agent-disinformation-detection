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

## 7. Proposed text preprocessing (proposal, not applied; to be agreed with the team)
- Drop empty texts and duplicates.
- Handle the "Reuters" signature (remove it, or compare results with and without it).
- Do not use `subject` or `date` as features.
- Lowercase, remove URLs, digits and punctuation; vectorize with word counts or TF-IDF.
- Split train/test before training, with the same ratio for both classes.
- Fix tokenization: keep apostrophes or drop single-letter tokens, to remove artifacts like `s`, `t`, `u`.

## 8. Suitability and limitations
- Suitable for a first Naive Bayes baseline (text column, two balanced classes).
- Limits: news articles, not short social-media posts; the model may learn the source or style instead of the truth.