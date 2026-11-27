"""Interface de l'Agent 1 pour le reste du projet.

Usage dans le code (ex. Agent 3) :
    from agents.agent1_naive_bayes.src.predict import score_text
    p = score_text("Some article text ...")      # float entre 0 et 1 = P(fake)

Usage en terminal (démo), depuis le dossier agents/agent1_naive_bayes :
    python -m src.predict "Texte de l'article ..."
    python -m src.predict --file article.txt --explain
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from .preprocess import MODEL_PATH, clean_article

_BUNDLE = None  # le modèle est chargé une seule fois, puis gardé en mémoire


def load_model(path: Path = MODEL_PATH) -> dict:
    """Charge models/nb_fake_news.joblib (vectorizer + Naive Bayes + calibration)."""
    global _BUNDLE
    if _BUNDLE is None:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {path}. Lance d'abord : python -m src.train"
            )
        _BUNDLE = joblib.load(path)
    return _BUNDLE


def _prepare(text: str, title: str) -> str:
    return clean_article(title, text) 


def _logit(nb, X) -> np.ndarray:
    """Score brut : log P(fake|texte) - log P(true|texte)."""
    lp = nb.predict_log_proba(X)
    return lp[:, 1] - lp[:, 0]


def score_texts(texts: list[str], titles: list[str] | None = None) -> list[float]:
    """Probabilité fake (0 à 1) pour une liste d'articles."""
    b = load_model()
    titles = titles or [""] * len(texts)
    cleaned = [_prepare(t, ti) for t, ti in zip(texts, titles)]
    X = b["vectorizer"].transform(cleaned)
    proba = b["calibrator"].predict_proba(_logit(b["nb"], X).reshape(-1, 1))[:, 1]
    # Texte vide ou sans aucun mot connu : aucune information -> 0.5 (indécis).
    no_info = np.asarray(X.sum(axis=1)).ravel() == 0
    proba[no_info] = 0.5
    return [float(p) for p in proba]


def score_text(text: str, title: str = "") -> float:
    """Probabilité que l'article soit fake : 0.0 = sûrement vrai, 1.0 = sûrement fake.

    Renvoie 0.5 si le texte ne contient aucun mot que le modèle connaît.
    Le modèle est entraîné sur des ARTICLES de presse anglais (pas des tweets).
    """
    return score_texts([text], [title])[0]


def explain_text(text: str, title: str = "", n: int = 8) -> dict:
    """Mots de l'article qui poussent le plus vers fake / vers true (pour la démo)."""
    b = load_model()
    vec, nb = b["vectorizer"], b["nb"]
    x = vec.transform([_prepare(text, title)])
    words = np.array(vec.get_feature_names_out())
    weight = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    cols = x.indices
    contrib = x.data * weight[cols]
    order = np.argsort(contrib)
    return {
        "vers_fake": [(str(words[cols[i]]), round(float(contrib[i]), 3))
                      for i in order[::-1][:n] if contrib[i] > 0],
        "vers_true": [(str(words[cols[i]]), round(float(contrib[i]), 3))
                      for i in order[:n] if contrib[i] < 0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score fake news d'un article")
    parser.add_argument("text", nargs="?", help="texte de l'article")
    parser.add_argument("--file", type=Path, help="lire l'article depuis un fichier .txt")
    parser.add_argument("--title", default="")
    parser.add_argument("--explain", action="store_true", help="afficher les mots décisifs")
    args = parser.parse_args()

    text = args.file.read_text(encoding="utf-8") if args.file else args.text
    if not text:
        parser.error("donne un texte ou --file")
    p = score_text(text, args.title)
    print(f"P(fake) = {p:.3f}  ->  {'FAKE' if p >= 0.5 else 'TRUE'}")
    if args.explain:
        print(explain_text(text, args.title))


if __name__ == "__main__":
    main()