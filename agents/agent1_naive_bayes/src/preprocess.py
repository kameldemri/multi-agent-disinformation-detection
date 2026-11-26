"""Prétraitement de l'Agent 1 (Naive Bayes, fake/true news).

Ce fichier contient TOUT ce qui prépare les données avant le modèle :
    A. chemins du projet
    B. construction du dataset (label, nettoyage des lignes, doublons)
    C. découpage train / validation / test (sans fuite)
    D. nettoyage du texte (signatures de source, apostrophes...)
    E. vectorisation (texte -> nombres)

La même fonction clean_text doit servir à l'entraînement (train.py) ET à la
prédiction (predict.py). Sinon le modèle voit un texte différent de celui sur
lequel il a appris.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.model_selection import StratifiedGroupKFold

# ===========================================================================
# A. Chemins
# ===========================================================================
# Ce fichier : <projet>/agents/agent1_naive_bayes/src/preprocess.py
#   parents[0] = src, parents[1] = agent1_naive_bayes, parents[3] = racine du projet
AGENT_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = ROOT / "data" / "raw" / "fake_news"          # fake.csv et true.csv
MODEL_PATH = AGENT_DIR / "models" / "nb_fake_news.joblib"
RESULTS_DIR = AGENT_DIR / "results"

RANDOM_STATE = 42   # graine fixe : mêmes résultats à chaque exécution
MIN_WORDS = 10      # en dessous, le texte est considéré comme vide/inutilisable
LABEL_FAKE = 1      # classe "positive" : celle qu'on veut détecter
LABEL_TRUE = 0


# ===========================================================================
# B. Construction du dataset
# ===========================================================================
def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lit fake.csv et true.csv (les fichiers originaux ne sont jamais modifiés)."""
    fake = pd.read_csv(Path(raw_dir) / "fake.csv")
    true = pd.read_csv(Path(raw_dir) / "true.csv")
    return fake, true


def normalize_key(s: str) -> str:
    """Version "comparable" d'un texte : minuscules, sans ponctuation, espaces simples.

    Sert à détecter les doublons même si deux textes diffèrent seulement par un
    espace, une majuscule ou un signe de ponctuation.
    """
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def build_dataset(
    raw_dir: Path = RAW_DIR,
    min_words: int = MIN_WORDS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict]:
    """Construit le dataset propre.

    Retourne (df, report) :
        df      colonnes : content, title, text, label, group
        report  dictionnaire de comptages (à citer dans le rapport)

    On N'UTILISE PAS `subject` ni `date` : ils trahissent la classe
    (true n'a que 2 subjects, fake en a 6 autres ; les périodes diffèrent).
    """
    fake, true = load_raw(raw_dir)

    # --- label : les fichiers n'en ont pas, on le déduit du nom du fichier ---
    fake["label"] = LABEL_FAKE
    true["label"] = LABEL_TRUE
    df = pd.concat([fake, true], ignore_index=True)
    for col in ("title", "text"):
        df[col] = df[col].fillna("").astype(str).str.strip()

    report: dict = {
        "lignes_brutes": int(len(df)),
        "fake_brut": int(len(fake)),
        "true_brut": int(len(true)),
    }

    # --- textes vides ou trop courts ------------------------------------------
    # 630 textes vides dans fake contre 1 dans true : "texte vide => fake" serait
    # une fuite de données, donc on supprime ces lignes.
    n_words = df["text"].str.split().str.len().fillna(0)
    short = n_words < min_words
    report["textes_courts_supprimes_fake"] = int((short & (df["label"] == LABEL_FAKE)).sum())
    report["textes_courts_supprimes_true"] = int((short & (df["label"] == LABEL_TRUE)).sum())
    df = df[~short].copy()

    # --- même texte avec deux labels différents --------------------------------
    # Un texte présent dans fake ET dans true est ambigu : on le retire des deux.
    df["_key"] = df["text"].map(normalize_key)
    n_labels = df.groupby("_key")["label"].transform("nunique")
    report["lignes_texte_dans_fake_et_true"] = int((n_labels > 1).sum())
    df = df[n_labels == 1].copy()

    # --- doublons sur le TEXTE (plus strict que df.duplicated() de l'EDA) ------
    before = len(df)
    df = df.drop_duplicates(subset="_key", keep="first").copy()
    report["doublons_texte_supprimes"] = int(before - len(df))

    # --- content = titre + texte ---------------------------------------------
    df["content"] = (df["title"] + " " + df["text"]).str.strip()

    # --- group : sert au split ---------------------------------------------
    # Deux articles avec le même titre sont presque toujours des quasi-doublons.
    # Ils doivent rester dans la MÊME partie (train ou test), sinon fuite.
    title_key = df["title"].map(normalize_key)
    row_id = pd.Series([f"__row{i}" for i in df.index], index=df.index)
    df["group"] = title_key.where(title_key != "", row_id)

    # --- mélange (sinon : tous les fake puis tous les true) -------------------
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    report["lignes_finales"] = int(len(df))
    report["fake_final"] = int((df["label"] == LABEL_FAKE).sum())
    report["true_final"] = int((df["label"] == LABEL_TRUE).sum())
    report["part_fake_pct"] = round(100 * report["fake_final"] / report["lignes_finales"], 1)

    return df[["content", "title", "text", "label", "group"]], report


# ===========================================================================
# C. Découpage train / validation / test
# ===========================================================================
def split_indices(
    df: pd.DataFrame, random_state: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Découpe ~70 % train / ~15 % validation / ~15 % test.

    - stratifié : même proportion fake/true dans chaque partie
    - par groupes : tous les articles ayant le même titre restent ensemble
    Retourne trois tableaux de POSITIONS de lignes (à utiliser avec .iloc).
    """
    y = df["label"].to_numpy()
    groups = df["group"].to_numpy()
    idx = np.arange(len(df))

    # 7 plis -> un pli = 1/7 = 14,3 % : c'est notre test.
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=random_state)
    rest, test = next(outer.split(idx, y, groups))

    # Sur le reste (85,7 %), 6 plis -> un pli = 14,3 % du total : c'est la validation.
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=random_state)
    tr, va = next(inner.split(rest, y[rest], groups[rest]))
    train, val = rest[tr], rest[va]

    # Vérification de sécurité : aucun groupe commun entre deux parties.
    g_train, g_val, g_test = set(groups[train]), set(groups[val]), set(groups[test])
    assert not (g_train & g_val), "fuite train/validation (groupes communs)"
    assert not (g_train & g_test), "fuite train/test (groupes communs)"
    assert not (g_val & g_test), "fuite validation/test (groupes communs)"
    return train, val, test


# ===========================================================================
# D. Nettoyage du texte
# ===========================================================================
# D1. Signatures de source. Chaque motif est une expression régulière (regex).
# L'ordre compte : le préfixe "VILLE (Reuters) -" doit partir AVANT qu'on retire
# le simple mot "Reuters".
_DASHES = "-\u2013\u2014"  # tiret, demi-cadratin, cadratin

SIGNATURE_PATTERNS: list[tuple[str, re.Pattern]] = [
    # "WASHINGTON (Reuters) - " au début : ville (<= 80 caractères) + (Reuters) + tiret.
    ("reuters_prefix", re.compile(rf"^\s*[^()\n]{{0,80}}?\(\s*reuters\s*\)\s*[{_DASHES}:]*\s*", re.I)),
    # Pied de page Reuters : "Our Standards: The Thomson Reuters Trust Principles."
    ("our_standards", re.compile(r"our standards:.*", re.I | re.S)),
    # Crédits de fin : "Reporting by X; Editing by Y"
    ("credits", re.compile(r"\(?\b(?:additional\s+)?(?:reporting|writing|editing)\s+by\b[^.\n]{0,150}\)?", re.I)),
    # Toute autre mention : "told Reuters", "Reuters/Ipsos", "Thomson Reuters"
    ("reuters_mention", re.compile(r"\b(?:thomson\s+)?reuters\b", re.I)),
        # Code JavaScript de l'embed Facebook collé à la fin de beaucoup d'articles fake :
    # "(function(d, s, id) { var js, fjs = ... }(document, 'script', 'facebook-jssdk'));"
    # Trouvé avec la cellule show_context de l'EDA (mots fjs, cdata, getelementbyid...).
    # Version complète du bloc (si on retrouve sa fin)...
    ("js_facebook_sdk", re.compile(
        r"\(function\s*\(\s*d\s*,\s*s\s*,\s*id\s*\).{0,1200}?facebook-jssdk['\"]?\s*\)*\s*;?",
        re.I | re.S)),
    # ... et morceaux isolés (si le bloc est coupé ou modifié).
    ("js_fragments", re.compile(
        r"//\s*<\s*!\[CDATA\[|//\s*\]\]>"
        r"|\(function\s*\(\s*d\s*,\s*s\s*,\s*id\s*\)\s*\{"
        r"|\bvar\s+js\s*,\s*fjs\s*=\s*d\.getElementsByTagName\(s\)\[0\]\s*;?"
        r"|if\s*\(\s*d\.getElementById\(\s*id\s*\)\s*\)\s*return\s*;?"
        r"|js\s*=\s*d\.createElement\(s\)\s*;?"
        r"|js\.id\s*=\s*id\s*;?"
        r"|js\.src\s*=\s*[^;]*;?"
        r"|fjs\.parentNode\.insertBefore\(\s*js\s*,\s*fjs\s*\)\s*;?",
        re.I)),
    # Promo d'un site fake : "Tune in to the Alternate Current Radio Network (ACR) ... The Boiler Room".
    ("acr_network", re.compile(r"\balternate current radio(?: network)?\b|\(ACR\)", re.I)),
    # Sensible à la casse (pas de re.I) pour ne pas toucher l'expression normale "boiler room tactics".
    ("boiler_room_show", re.compile(r"\bACR(?:'s| s)?\b|\bTHE BOILER ROOM\b|\bBOILER ROOM\b|\bThe Boiler Room\b")),
    # Nom du rédacteur de 21st Century Wire (signature d'un site fake).
    ("henningsen", re.compile(r"\bpatrick\s+henningsen\b|\bhenningsen\b", re.I)),
    # Signatures typiques des sites "fake" : à confirmer avec les cellules de
    # l'EDA (marqueurs, mots déséquilibrés) ; n'en garder que celles qui sont
    # très déséquilibrées entre classes.
    ("pic_twitter", re.compile(r"pic\.twitter\.com/\S+", re.I)),
    ("featured_image", re.compile(r"featured\s+image(?:\s+(?:via|credit|by)\b[^.\n]{0,80})?", re.I)),
    ("image_via", re.compile(r"\b(?:image|photo|screenshot|video)\s+(?:via|by|credit|courtesy of)\b[^.\n]{0,80}", re.I)),
    ("getty", re.compile(r"\bgetty\s+images\b", re.I)),
    ("21cw", re.compile(r"21st\s+century\s+wire(?:\s+says)?", re.I)),
    ("read_more", re.compile(r"\bread\s+more\b:?", re.I)),
    ("watch_below", re.compile(r"\bwatch\s+(?:the\s+)?(?:video|clip)\s+below\b:?", re.I)),
    # Liens complets (http://..., www....)
    ("url", re.compile(r"https?://\S+|www\.\S+", re.I)),
    # Liens sans http : "tmsnrt.rs/2Ge", "twitter.com/xyz", "youtube.com/watch..."
    ("short_link", re.compile(r"\b[\w-]+\.(?:rs|ly|co|com|org|net)/\S*", re.I)),
    # Mots de "crédit média" trouvés avec la cellule des mots déséquilibrés de l'EDA
    # (0 % dans true, 2 à 17 % dans fake) + horodatage Reuters "GMT".
    ("media_words", re.compile(
        r"\b(?:flickr|youtube|youtu|screengrab|screenshot|subscribe|somodevilla|filessupport|gmt)\b",
        re.I)),
]


def strip_source_signatures(text: str) -> str:
    """Supprime les signatures de source (Reuters, crédits, liens...)."""
    for _name, pattern in SIGNATURE_PATTERNS:
        text = pattern.sub(" ", text)
    return text


# D2. Normalisation : minuscules, apostrophes, ponctuation.
# Apostrophes "typographiques" -> apostrophe simple '
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "`": "'", "\u00b4": "'"})

_US = re.compile(r"\bu\.s\.(?:a\.)?", re.I)          # U.S. / U.S.A. -> usa
_MENTION = re.compile(r"@\w+")                        # @utilisateur
_CANT = re.compile(r"\bcan't\b")
_WONT = re.compile(r"\bwon't\b")
_NT = re.compile(r"(?<=\w)n't\b")                     # don't -> do not
_SUFFIX = re.compile(r"'(?:s|re|ve|ll|d|m)\b")        # 's 're 've 'll 'd 'm -> supprimés
_NOT_LETTERS = re.compile(r"[^a-z]+")                 # tout sauf a-z -> espace


def normalize_text(text: str) -> str:
    """Minuscules, apostrophes, "U.S." -> "usa", puis uniquement des lettres a-z."""
    t = text.translate(_APOSTROPHES)
    t = _US.sub(" usa ", t)   # avant lower() et avant la ponctuation
    t = t.lower()
    t = _MENTION.sub(" ", t)
    t = _CANT.sub("cannot", t)
    t = _WONT.sub("will not", t)
    t = _NT.sub(" not", t)    # "don't" -> "do not" (plus de token "t" isolé)
    t = _SUFFIX.sub("", t)    # "trump's" -> "trump" (plus de token "s" isolé)
    t = _NOT_LETTERS.sub(" ", t)
    return t.strip()


def clean_text(text: str, remove_signatures: bool = True) -> str:
    """Nettoyage complet d'un texte (utilisé partout : entraînement + prédiction)."""
    if not isinstance(text, str):
        return ""
    if remove_signatures:
        text = strip_source_signatures(text)
    return normalize_text(text)


def clean_series(texts: Iterable[str], remove_signatures: bool = True) -> pd.Series:
    """Applique clean_text à toute une colonne."""
    texts = pd.Series(texts)
    return pd.Series([clean_text(t, remove_signatures) for t in texts], index=texts.index)


def clean_article(title: str, text: str, remove_signatures: bool = True) -> str:
    """Nettoie le titre et le texte SÉPARÉMENT, puis les colle."""
    return f"{clean_text(title, remove_signatures)} {clean_text(text, remove_signatures)}".strip()


def clean_articles(titles, texts, remove_signatures: bool = True) -> pd.Series:
    titles = pd.Series(titles)
    texts = pd.Series(texts, index=titles.index)
    return pd.Series(
        [clean_article(ti, tx, remove_signatures) for ti, tx in zip(titles, texts)],
        index=titles.index,
    )
# ===========================================================================
# E. Vectorisation (texte -> nombres)
# ===========================================================================
# Un "token" = au moins 3 lettres a-z. Cela élimine directement les tokens
# de 1-2 lettres (s, t, u, it, of...) qui venaient des apostrophes.
TOKEN_PATTERN = r"(?u)\b[a-z]{3,}\b"

# Stopwords anglais de sklearn, restreints aux mots de >= 3 lettres
# (les autres sont déjà éliminés par TOKEN_PATTERN).
STOP_WORDS = sorted(w for w in ENGLISH_STOP_WORDS if len(w) >= 3)


def build_vectorizer(
    kind: str = "tfidf",
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int = 5,
    max_df: float = 1.0,
    stop_words: str | None = None,
    max_features: int | None = 200_000,
):
    """Crée le vectorizer (transforme un texte en vecteur de nombres).

    kind          "tfidf" (poids TF-IDF) ou "count" (simple comptage de mots)
    ngram_range   (1,1) = mots seuls ; (1,2) = mots + paires de mots
    min_df        ignore les mots présents dans moins de min_df articles
    max_df        ignore les mots présents dans plus de max_df (proportion) des articles
    stop_words    None ou "english"
    max_features  garde seulement les N mots les plus fréquents (taille du modèle)
    """
    common = dict(
        lowercase=False,  # le texte est déjà en minuscules (clean_text)
        token_pattern=TOKEN_PATTERN,
        ngram_range=tuple(ngram_range),
        min_df=min_df,
        max_df=max_df,
        stop_words=STOP_WORDS if stop_words == "english" else None,
        max_features=max_features,
    )
    if kind == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, **common)
    if kind == "count":
        return CountVectorizer(**common)
    raise ValueError(f"kind inconnu : {kind!r}")