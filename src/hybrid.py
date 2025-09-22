# src/hybrid.py
from pathlib import Path
import pandas as pd
from src import utils
import joblib

def main():
    in_path = utils.processed_dir() / "combined_lexicon.csv"
    if not in_path.exists():
        print("Missing combined_lexicon.csv")
        return
    df = pd.read_csv(in_path, low_memory=False)

    model_path = Path(utils.ROOT) / "models" / "tfidf_lr.joblib"
    if not model_path.exists():
        print("Model not found. Run training first.")
        return
    pipe = joblib.load(model_path)

    X = df['text_clean'].fillna("").astype(str)
    probs = pipe.predict_proba(X)
    max_conf = probs.max(axis=1)
    ml_pred = pipe.predict(X)

    df['ml_pred'] = ml_pred
    df['ml_conf'] = max_conf
    df['rating'] = pd.to_numeric(df.get('rating', None), errors='coerce').fillna(0)

    def hybrid_decision(vader_label, ml_pred, ml_conf, rating):
        # simple rule-based ensembling: take ML if confident, else fallback to vader (but respect high rating)
        if ml_conf >= 0.80:
            return ml_pred
        if rating >= 4:
            return 'positive' if vader_label != 'negative' else 'neutral'
        return vader_label

    df['hybrid_label'] = df.apply(lambda r: hybrid_decision(r['vader_label'], r['ml_pred'], r['ml_conf'], r['rating']), axis=1)
    out = utils.processed_dir() / "combined_hybrid.csv"
    df.to_csv(out, index=False)
    print("Saved hybrid results to", out)

if __name__ == "__main__":
    main()
