"""Entraînement, évaluation et expériences de l'Agent 1 (Naive Bayes fake news).

Lancer depuis le dossier agents/agent1_naive_bayes :
    python -m src.train                  # entraînement complet
    python -m src.train --quick          # test rapide (petite grille)
    python -m src.train --experiments    # expériences pour le rapport (après l'entraînement)

Ce fichier contient :
    A. le modèle : Naive Bayes, score brut, calibration
    B. l'évaluation : métriques, graphiques, mots discriminants
    C. le choix des réglages (grille sur la validation)
    D. l'entraînement complet (fonction run)
    E. les expériences du rapport (fonction run_experiments)
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime
from pathlib import Path
import joblib 
import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : on enregistre les images dans des fichiers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from .preprocess import (
    MODEL_PATH,
    RAW_DIR,
    RESULTS_DIR,
    build_dataset,
    build_vectorizer,
    clean_articles,
    clean_series,
    split_indices,
)

ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0]


# ===========================================================================
# A. Le modèle
# ===========================================================================
def build_nb(kind: str = "multinomial", alpha: float = 1.0):
    """Crée le modèle Naive Bayes. alpha = lissage de Laplace."""
    if kind == "multinomial":
        return MultinomialNB(alpha=alpha)
    if kind == "complement":
        return ComplementNB(alpha=alpha)
    raise ValueError(f"nb_kind inconnu : {kind!r}")


def nb_logit(nb, X) -> np.ndarray:
    """Score brut = log P(fake|texte) - log P(true|texte).

    > 0 : le modèle penche pour fake ; < 0 : pour true.
    Marche pour MultinomialNB et ComplementNB.
    """
    log_proba = nb.predict_log_proba(X)
    return log_proba[:, 1] - log_proba[:, 0]


def fit_model(texts, y, params: dict):
    """Apprend vectorizer + Naive Bayes SUR LES DONNÉES D'ENTRAÎNEMENT SEULEMENT."""
    vec = build_vectorizer(
        kind=params["vectorizer"],
        ngram_range=tuple(params["ngram_range"]),
        min_df=params.get("min_df", 5),
        max_df=params.get("max_df", 1.0),
        stop_words=params.get("stop_words"),
        max_features=params.get("max_features", 200_000),
    )
    X = vec.fit_transform(texts)
    nb = build_nb(params["nb_kind"], params["alpha"]).fit(X, y)
    return vec, nb


def fit_calibrator(vec, nb, texts, y) -> LogisticRegression:
    """Calibration de Platt.

    Naive Bayes donne des probabilités extrêmes (0.999 ou 0.001), souvent
    exagérées. On apprend, sur la validation, une petite régression logistique
    qui transforme le score brut en vraie probabilité (0.7 ~ 70 % de chances).
    """
    logit = nb_logit(nb, vec.transform(texts)).reshape(-1, 1)
    return LogisticRegression(C=1e4, max_iter=1000).fit(logit, y)


def predict_proba_fake(vec, nb, calibrator, texts) -> np.ndarray:
    """Probabilité que chaque texte (déjà nettoyé) soit fake."""
    X = vec.transform(texts)
    if calibrator is None:
        return nb.predict_proba(X)[:, 1]
    return calibrator.predict_proba(nb_logit(nb, X).reshape(-1, 1))[:, 1]


# ===========================================================================
# B. Évaluation
# ===========================================================================
def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Métriques de classification. La classe positive est fake (label 1).

    precision : parmi les articles prédits fake, combien le sont vraiment
    recall    : parmi les vrais fake, combien on en a trouvé
    brier     : qualité des probabilités (0 = parfait ; plus bas = mieux)
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def save_confusion_matrix(y_true, proba, path: Path, threshold: float = 0.5) -> None:
    pred = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["prédit true", "prédit fake"])
    ax.set_yticks([0, 1], ["vrai true", "vrai fake"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Matrice de confusion (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_roc_curve(y_true, proba, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, proba):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs (recall)")
    ax.set_title("Courbe ROC (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_calibration_plot(y_true, proba_raw, proba_cal, path: Path) -> None:
    """Courbe de fiabilité : si la courbe suit la diagonale, les probabilités sont honnêtes."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for proba, name in [(proba_raw, "NB brut"), (proba_cal, "NB calibré")]:
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label=name)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("Probabilité prédite (fake)")
    ax.set_ylabel("Proportion réelle de fake")
    ax.set_title("Calibration (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_words(vec, nb, n: int = 20) -> dict:
    """Mots les plus discriminants.
    Pour chaque mot : log P(mot|fake) - log P(mot|true).
    Grand positif -> mot typique des fake ; grand négatif -> typique des true.
    """
    words = np.array(vec.get_feature_names_out())
    diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    order = np.argsort(diff)
    return {
        "fake": [(str(words[i]), round(float(diff[i]), 3)) for i in order[::-1][:n]],
        "true": [(str(words[i]), round(float(diff[i]), 3)) for i in order[:n]],
    }
def print_data_report(r: dict) -> None:
    """Affiche le bilan du nettoyage du dataset sous forme lisible."""
    print(f"  Lignes au départ            : {r['lignes_brutes']:>7d}  (fake {r['fake_brut']}, true {r['true_brut']})")
    print(f"  - textes trop courts (fake) : {r['textes_courts_supprimes_fake']:>7d}")
    print(f"  - textes trop courts (true) : {r['textes_courts_supprimes_true']:>7d}")
    print(f"  - texte dans fake ET true   : {r['lignes_texte_dans_fake_et_true']:>7d}")
    print(f"  - doublons sur le texte     : {r['doublons_texte_supprimes']:>7d}")
    print(f"  Lignes à la fin             : {r['lignes_finales']:>7d}  "
          f"(fake {r['fake_final']}, true {r['true_final']}, {r['part_fake_pct']} % fake)")


def print_split(split_info: dict) -> None:
    """Affiche la taille des 3 groupes."""
    print(f"  {'groupe':<12}{'articles':>10}{'% fake':>10}")
    for name, label in [("train", "train"), ("val", "validation"), ("test", "test")]:
        d = split_info[name]
        print(f"  {label:<12}{d['n']:>10d}{d['part_fake_pct']:>9.1f}%")


def print_metrics(title: str, m: dict) -> None:
    """Affiche les métriques et la matrice de confusion sous forme de tableau."""
    c = m["confusion"]
    print(f"\n=== {title} ===")
    print(f"  accuracy  : {m['accuracy']:.1%}   (articles bien classés)")
    print(f"  precision : {m['precision']:.1%}   (quand il dit 'fake', il a raison)")
    print(f"  recall    : {m['recall']:.1%}   (part des vrais fake qu'il retrouve)")
    print(f"  F1        : {m['f1']:.3f}")
    print(f"  ROC-AUC   : {m['roc_auc']:.3f}   (1 = parfait, 0.5 = hasard)")
    print(f"  Brier     : {m['brier']:.4f}  (plus bas = mieux)")
    print("\n  Matrice de confusion :")
    print("                    prédit TRUE   prédit FAKE")
    print(f"    vrai TRUE      {c['tn']:>11d}   {c['fp']:>11d}")
    print(f"    vrai FAKE      {c['fn']:>11d}   {c['tp']:>11d}")


# ===========================================================================
# C. Choix des réglages (grille, sur la VALIDATION uniquement)
# ===========================================================================
def grid_search(texts_tr, y_tr, texts_va, y_va, quick: bool = False) -> pd.DataFrame:
    """Teste plusieurs combinaisons et les note sur la validation (jamais sur le test).

    Pour chaque réglage du vectorizer on ne vectorise qu'UNE fois, puis on
    essaie tous les (modèle NB, alpha) : c'est beaucoup plus rapide.
    """
    if quick:
        vec_grid = [("tfidf", (1, 1), None), ("count", (1, 1), None)]
        alphas = [0.1, 1.0]
    else:
        vec_grid = list(itertools.product(["tfidf", "count"], [(1, 1), (1, 2)], [None, "english"]))
        alphas = ALPHAS

    rows = []
    for kind, ngram, stop in vec_grid:
        t0 = time.time()
        vec = build_vectorizer(kind=kind, ngram_range=ngram, stop_words=stop)
        X_tr = vec.fit_transform(texts_tr)      # le vocabulaire vient du TRAIN seulement
        X_va = vec.transform(texts_va)
        for nb_kind, alpha in itertools.product(["multinomial", "complement"], alphas):
            nb = build_nb(nb_kind, alpha).fit(X_tr, y_tr)
            score = nb_logit(nb, X_va)
            rows.append({
                "vectorizer": kind, "ngram_range": ngram, "stop_words": stop,
                "nb_kind": nb_kind, "alpha": alpha,
                "val_f1": f1_score(y_va, (score > 0).astype(int)),
                "val_roc_auc": roc_auc_score(y_va, score),
                "n_features": X_tr.shape[1],
            })
        print(f"  vectorizer={kind:5s} ngram={ngram} stop={stop!s:7s} "
              f"-> {X_tr.shape[1]:>7d} mots ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(rows)
    return res.sort_values(["val_f1", "val_roc_auc"], ascending=False).reset_index(drop=True)


def row_to_params(row: pd.Series) -> dict:
    return {
        "vectorizer": row["vectorizer"],
        "ngram_range": list(row["ngram_range"]),
        "stop_words": None if pd.isna(row["stop_words"]) else row["stop_words"],
        "nb_kind": row["nb_kind"],
        "alpha": float(row["alpha"]),
        "min_df": 5,
        "max_df": 1.0,
        "max_features": 200_000,
    }


# ===========================================================================
# D. Entraînement complet
# ===========================================================================
def run(raw_dir: Path, model_path: Path, results_dir: Path, quick: bool) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 1. Dataset ---------------------------------------------------------------
    print("[1/7] Construction du dataset...")
    df, data_report = build_dataset(raw_dir)
    print_data_report(data_report)

    # 2. Nettoyage du texte ---------------------------------------------------
    print("[2/7] Nettoyage du texte (Reuters, apostrophes...)...")
    df["clean"] = clean_articles(df["title"], df["text"])
    # Après nettoyage, certains textes peuvent être devenus presque vides : on les retire.
    keep = df["clean"].str.split().str.len().fillna(0) >= 5
    data_report["lignes_vides_apres_nettoyage"] = int((~keep).sum())
    df = df[keep].reset_index(drop=True)

    # 3. Split ------------------------------------------------------------------
    print("[3/7] Découpage train / validation / test...")
    tr, va, te = split_indices(df)
    X, y = df["clean"], df["label"]
    split_info = {
        name: {"n": int(len(idx)), "part_fake_pct": round(100 * float(y.iloc[idx].mean()), 1)}
        for name, idx in [("train", tr), ("val", va), ("test", te)]
    }
    print_split(split_info)
    # 4. Choix des réglages (validation) --------------------------------------
    print("[4/7] Recherche des meilleurs réglages (sur la validation)...")
    results = grid_search(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], quick=quick)
    results.to_csv(results_dir / "grid_results.csv", index=False)
    print(results.head(5).to_string())
    params = row_to_params(results.iloc[0])
    print("Meilleurs réglages :", params)

    # 5. Modèle final sur le train -----------------------------------------------
    print("[5/7] Entraînement final...")
    vec, nb = fit_model(X.iloc[tr], y.iloc[tr], params)

    # 6. Calibration (validation) --------------------------------------------------
    print("[6/7] Calibration des probabilités...")
    calibrator = fit_calibrator(vec, nb, X.iloc[va], y.iloc[va])

    # 7. Évaluation finale sur le TEST ---------------------------------------------
    print("[7/7] Évaluation sur le test...")
    y_te = y.iloc[te].to_numpy()
    p_raw = predict_proba_fake(vec, nb, None, X.iloc[te])
    p_cal = predict_proba_fake(vec, nb, calibrator, X.iloc[te])
    metrics = {"brut": compute_metrics(y_te, p_raw), "calibre": compute_metrics(y_te, p_cal)}
    print_metrics("TEST - probabilités brutes", metrics["brut"])
    print_metrics("TEST - probabilités calibrées (utilisées par l'Agent 3)", metrics["calibre"])

    save_confusion_matrix(y_te, p_cal, results_dir / "confusion_matrix.png")
    save_roc_curve(y_te, p_cal, results_dir / "roc_curve.png")
    save_calibration_plot(y_te, p_raw, p_cal, results_dir / "calibration.png")
    words = top_words(vec, nb, n=25)
    print("Top mots fake :", [w for w, _ in words["fake"]][:15])
    print("Top mots true :", [w for w, _ in words["true"]][:15])

    # Sauvegarde ---------------------------------------------------------------------
    # stop_words_ peut contenir des centaines de milliers de mots écartés : on le
    # vide pour que le fichier .joblib reste petit (important pour GitHub).
    if hasattr(vec, "stop_words_"):
        vec.stop_words_ = None
    bundle = {
        "vectorizer": vec,
        "nb": nb,
        "calibrator": calibrator,
        "meta": {
            "params": params,
            "metrics_test": metrics,
            "label_map": {"fake": 1, "true": 0},
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "n_train": int(len(tr)),
        },
    }
    joblib.dump(bundle, model_path, compress=3)
    print(f"Modèle sauvegardé : {model_path} ({model_path.stat().st_size / 1e6:.1f} Mo)")

    summary = {
        "data_report": data_report,
        "split": split_info,
        "best_params": params,
        "metrics_test": metrics,
        "top_words": words,
        "duree_secondes": round(time.time() - t_start),
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    df["clean"] = clean_articles(df["title"], df["text"])
    print(f"Rapport : {results_dir / 'summary.json'}")
    return summary


# ===========================================================================
# E. Expériences pour le rapport
# ===========================================================================
def _train_and_score(texts, y, tr, te, params) -> dict:
    """Entraîne sur train, évalue sur test (probabilités brutes, sans calibration)."""
    vec, nb = fit_model(texts.iloc[tr], y.iloc[tr], params)
    proba = predict_proba_fake(vec, nb, None, texts.iloc[te])
    m = compute_metrics(y.iloc[te], proba)
    return {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}


def run_experiments(raw_dir: Path, model_path: Path, results_dir: Path) -> pd.DataFrame:
    """A. fuite Reuters (avec/sans signatures + règle bête) ; B. title / text / les deux.

    Prérequis : avoir lancé l'entraînement (on réutilise les meilleurs réglages).
    """
    params = joblib.load(model_path)["meta"]["params"]
    df, _ = build_dataset(raw_dir)

    # Même nettoyage et même filtre que run(), sinon le split n'est pas identique.
    df["clean"] = clean_articles(df["title"], df["text"])
    df = df[df["clean"].str.split().str.len().fillna(0) >= 5].reset_index(drop=True)

    tr, _va, te = split_indices(df)   # même split que run() (graine fixe)
    y = df["label"]
    rows = []

    print("A1. Avec signatures (Reuters gardé) ...")
    raw_clean = clean_articles(df["title"], df["text"], remove_signatures=False)
    rows.append({"experience": "A1 avec signatures (Reuters gardé)",
                 **_train_and_score(raw_clean, y, tr, te, params)})

    print("A2. Sans signatures (Reuters retiré) ...")
    rows.append({"experience": "A2 sans signatures (Reuters retiré)",
                 **_train_and_score(df["clean"], y, tr, te, params)})

    # Règle bête : "le mot Reuters apparaît => true". Aucun apprentissage.
    has_reuters = df["content"].str.contains("Reuters", case=False, regex=False)
    pred_fake = (~has_reuters).astype(float)
    m = compute_metrics(y.iloc[te], pred_fake.iloc[te])
    rows.append({"experience": "A3 règle bête 'contient Reuters => true'",
                 **{k: m[k] for k in ("accuracy", "precision", "recall", "f1")}, "roc_auc": None})

    print("B1 title seul ...")
    rows.append({"experience": "B1 title seul",
                 **_train_and_score(clean_series(df["title"]), y, tr, te, params)})

    print("B2 text seul ...")
    rows.append({"experience": "B2 text seul",
                 **_train_and_score(clean_series(df["text"]), y, tr, te, params)})

    print("B3 title + text ...")
    rows.append({"experience": "B3 title + text",
                 **_train_and_score(df["clean"], y, tr, te, params)})

    out = pd.DataFrame(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(results_dir / "experiments.csv", index=False)
    print(out.to_string(index=False))
    return out

# ===========================================================================
# Point d'entrée
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement Agent 1 (Naive Bayes fake news)")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--quick", action="store_true", help="petite grille (test rapide)")
    parser.add_argument("--experiments", action="store_true",
                        help="lancer les expériences du rapport (après l'entraînement)")
    args = parser.parse_args()
    if args.experiments:
        run_experiments(args.raw_dir, args.model_path, args.results_dir)
    else:
        run(args.raw_dir, args.model_path, args.results_dir, args.quick)


if __name__ == "__main__":
    main()