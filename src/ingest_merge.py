# src/ingest_merge.py
from pathlib import Path
import pandas as pd
from src import utils

def main():
    raw_files = utils.list_raw_csvs()
    print("Found raw files:", [p.name for p in raw_files])
    dfs = []
    for p in raw_files:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            print(f"Failed to read {p.name}: {e}")
            continue
        df['source'] = p.stem
        dfs.append(df)

    if not dfs:
        print("No CSVs found in data/raw. Put your dataset files there and re-run.")
        return

    combined = pd.concat(dfs, ignore_index=True, sort=False)

    # try to normalize a text column to 'text'
    if 'text' not in combined.columns:
        for cand in ['review_text','review','reviewText','content','body','comment']:
            if cand in combined.columns:
                combined = combined.rename(columns={cand: 'text'})
                break
    if 'review_id' not in combined.columns:
        for cand in ['id','reviewId','review_id','reviewID']:
            if cand in combined.columns:
                combined = combined.rename(columns={cand: 'review_id'})
                break

    out = utils.processed_dir() / "combined_raw.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print("Saved combined_raw.csv at", out, "rows:", len(combined))

if __name__ == "__main__":
    main()
