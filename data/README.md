# Raw data

Raw datasets are downloaded from their original sources and are never committed.

## Cresci-2017 (Twitter social spambots vs genuine accounts)

- Source: Kaggle, https://www.kaggle.com/datasets/hashemalsalmi/cresci-2017
- Location: `data/raw/cresci_2017/`
- Expected layout (keep folder and file names exactly):

```
data/raw/cresci_2017/
├── genuine_accounts/
│   ├── tweets.csv
│   └── users.csv
└── social_spambots/
    ├── tweets.csv
    └── users.csv
```

- The whole `data/raw/cresci_2017/` directory is gitignored (see `.gitignore`); the ~965 MB
  `genuine_accounts/tweets.csv` must never be committed.
- Overview EDA: `analysis/eda_cresci_overview.ipynb`. Dataset notes: `docs/datasets/cresci-2017.md`.
- Other raw datasets use the same pattern: `data/raw/fake_news/`, `data/raw/twitter_bot_detection/`.
