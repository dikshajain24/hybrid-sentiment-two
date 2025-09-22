# src/train_tfidf_lr.py
from pathlib import Path
import pandas as pd
from src import utils
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

def main():
    in_path = utils.processed_dir() / "combined_lexicon.csv"
    if not in_path.exists():
        print("Missing combined_lexicon.csv — run lexicon_sentiment first.")
        return
    df = pd.read_csv(in_path, low_memory=False)

    # if you already have human labels, use them; otherwise use VADER as weak labels
    if 'label' in df.columns and df['label'].notna().sum() >= 50:
        train_df = df.dropna(subset=['label']).copy()
        X = train_df['text_clean'].astype(str)
        y = train_df['label'].astype(str)
        print("Using existing 'label' column for supervised training.")
    else:
        train_df = df.copy()
        train_df['label'] = train_df['vader_label']
        X = train_df['text_clean'].astype(str)
        y = train_df['label'].astype(str)
        print("No ground truth labels found (or too few). Using VADER labels as weak supervision.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print("Validation results:\n", classification_report(y_test, preds))

    models_dir = Path(utils.ROOT) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, models_dir / "tfidf_lr.joblib")
    print("Saved model at", models_dir / "tfidf_lr.joblib")

if __name__ == "__main__":
    main()
