# src/lexicon_sentiment.py
from pathlib import Path
import pandas as pd
from src import utils
import nltk

def main():
    in_path = utils.processed_dir() / "combined_clean.csv"
    if not in_path.exists():
        print("Missing combined_clean.csv — run preprocess first.")
        return
    # download vader lexicon (quiet)
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    df = pd.read_csv(in_path, low_memory=False)
    df['vader_compound'] = df['text_clean'].fillna("").apply(lambda t: sia.polarity_scores(str(t))['compound'])

    def vader_label(score):
        if score >= 0.05:
            return 'positive'
        if score <= -0.05:
            return 'negative'
        return 'neutral'

    df['vader_label'] = df['vader_compound'].apply(vader_label)
    out = utils.processed_dir() / "combined_lexicon.csv"
    df.to_csv(out, index=False)
    print("Saved combined_lexicon.csv at", out)

if __name__ == "__main__":
    main()
