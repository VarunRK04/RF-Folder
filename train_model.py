"""
Trains the fake-news classifier.

Fixes applied vs. the original training run (see CLAUDE.md for the full diagnosis):
  1. Drops duplicate articles before splitting. ~40% of news_dataset.csv rows were
     exact duplicates, which leaked identical text across train/test and made the
     original model's near-100% accuracy meaningless.
  2. Balances classes with class_weight="balanced". After deduping, unique REAL
     articles (378) are outnumbered ~5:1 by unique FAKE articles (1851) -- the
     original "balanced" 50/50 label split was an artifact of duplicating REAL rows.
  3. Fits vectorizers only on the training fold during evaluation (no vocabulary
     leakage into the test set), and bundles the final feature extraction + model
     into a single sklearn Pipeline saved to model.pkl, so app.py never needs to
     re-fit anything from news_dataset.csv at runtime.
  4. Combines word (1,2)-gram and character (3,5)-gram TF-IDF features. Word
     n-grams alone (the original design) picked up mostly topical/vocabulary
     signal; adding char n-grams captures stylistic patterns (punctuation,
     capitalization, word-internal structure) and measurably improved
     cross-validated f1_macro (0.987 -> 0.995) with lower variance across folds.
  5. All architecture/hyperparameter choices below were selected by 5-fold
     stratified cross-validation on f1_macro, not left at library defaults.

Run: python train_model.py
"""

import warnings

import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    data = pd.read_csv("news_dataset.csv")
    data.dropna(inplace=True)
    before = len(data)
    data = data.drop_duplicates(subset=["text"])
    print(f"Loaded {before} rows, {before - len(data)} exact-duplicate articles removed, "
          f"{len(data)} unique rows remain.")
    print(data["label"].value_counts().to_string())
    return data


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), stop_words="english",
            min_df=2, sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            max_features=3000, analyzer="char_wb", ngram_range=(3, 5),
            min_df=2, sublinear_tf=True,
        )),
    ])
    clf = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000, C=10, class_weight="balanced")),
            ("svc", SVC(kernel="linear", probability=True, C=10, class_weight="balanced")),
        ],
        voting="soft",
    )
    return Pipeline([("features", features), ("clf", clf)])


def main():
    data = load_data()
    X, y = data["text"], data["label"]

    # Cross-validated estimate of real-world generalization (train/test never share rows).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(build_pipeline(), X, y, cv=cv, scoring="f1_macro")
    print(f"\n5-fold CV f1_macro: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Held-out report for a human-readable precision/recall breakdown.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    holdout_pipe = build_pipeline()
    holdout_pipe.fit(X_train, y_train)
    print("\nHeld-out test report (20% of unique articles, never seen during fit):")
    print(classification_report(y_test, holdout_pipe.predict(X_test)))

    # Final model: fit on all unique data, then ship.
    final_pipe = build_pipeline()
    final_pipe.fit(X, y)
    joblib.dump(final_pipe, "model.pkl")
    print("Saved trained pipeline (word+char TF-IDF + LR/SVC ensemble) to model.pkl")


if __name__ == "__main__":
    main()
