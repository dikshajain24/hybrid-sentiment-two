# app.py — Hybrid Sentiment Analysis Dashboard (patched: safe chunked filters + friendly messages)

# --- STARTUP GUARD ---
import os, sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure src is a package
if not (ROOT / "src" / "__init__.py").exists():
    (ROOT / "src").mkdir(parents=True, exist_ok=True)
    (ROOT / "src" / "__init__.py").write_text("# auto-created\n")

# Create processed dir and demo CSV if empty
processed_dir = ROOT / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)
if not any(processed_dir.glob("*.csv")):
    demo_csv = processed_dir / "combined_demo.csv"
    demo_csv.write_text(
        "review_id,product_id,brand_name,product_title,review_text,review_rating,review_date,price,verified_purchases\n"
        "1,1001,DemoBrand,Demo Lipstick,\"Love this lipstick — color pops and lasts!\",5,2023-08-01,29.99,True\n"
        "2,1001,DemoBrand,Demo Lipstick,\"Not what I expected, too dry\",2,2023-07-15,29.99,False\n"
        "3,1002,DemoBrand,Demo Mascara,\"Okay for the price\",3,2023-06-10,15.00,True\n"
    )
    print("[startup] Demo CSV created:", demo_csv)

# Safe model loader
PIPE = None
MODEL_PATH = ROOT / "models" / "tfidf_lr.joblib"
try:
    import joblib
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        PIPE = joblib.load(MODEL_PATH)
        print("[startup] Model loaded")
    else:
        print("[startup] Model not found, continuing without ML model")
except Exception as e:
    print("[startup] Model load failed:", e)

st.session_state.setdefault("PIPE", PIPE)
st.session_state.setdefault("PROCESSED_DIR", str(processed_dir))

# Ensure NLTK Vader is available
try:
    import nltk
    nltk.data.find("sentiment/vader_lexicon.zip")
except Exception:
    try:
        nltk.download("vader_lexicon", quiet=True)
    except Exception as e:
        print("[startup] Failed to download vader lexicon:", e)
# --- END STARTUP GUARD ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Helpers ---

def safe_read_csv(path_or_buffer, max_rows_for_ui=250000):
    """Read *up to* max_rows_for_ui rows for interactive UI to avoid OOM."""
    try:
        if hasattr(path_or_buffer, "read"):
            # uploaded file-like
            try:
                path_or_buffer.seek(0)
            except Exception:
                pass
            df = pd.read_csv(path_or_buffer, low_memory=False)
            try:
                path_or_buffer.seek(0)
            except Exception:
                pass
            if len(df) > max_rows_for_ui:
                return df.sample(n=max_rows_for_ui, random_state=42).reset_index(drop=True)
            return df
        else:
            # filepath
            return pd.read_csv(path_or_buffer, nrows=max_rows_for_ui, low_memory=False)
    except Exception as e:
        print("[safe_read_csv] Failed:", e)
        return pd.DataFrame()

def safe_sample(df, n, random_state=42):
    if df is None or df.empty:
        return df.head(0)
    n_eff = min(max(0, int(n)), len(df))
    return df.sample(n=n_eff, random_state=random_state) if n_eff > 0 else df.head(0)

def ml_predict_safe(texts):
    """Wrap model prediction; returns (preds, confs)."""
    pipe = st.session_state.get("PIPE", None)
    if pipe is None:
        return [None]*len(texts), [0.0]*len(texts)
    try:
        preds = pipe.predict(texts)
        try:
            probs = pipe.predict_proba(texts)
            confs = list(probs.max(axis=1))
        except Exception:
            confs = [0.0]*len(texts)
        return preds, confs
    except Exception as e:
        print("[ml_predict_safe] failed:", e)
        return [None]*len(texts), [0.0]*len(texts)

def hybrid_label(row):
    vader = SentimentIntensityAnalyzer()
    text = row.get('text_clean', "") or ""
    score = vader.polarity_scores(text)['compound']
    label_vader = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"

    preds, confs = ml_predict_safe([text])
    label_ml, conf = preds[0], confs[0]

    # priority: confident ML, then rating heuristic, then vader
    try:
        rating = float(row.get('review_rating', np.nan))
    except Exception:
        rating = np.nan

    if label_ml is not None and conf >= 0.8:
        return label_ml
    if not np.isnan(rating) and rating >= 4 and label_vader != "negative":
        return "positive"
    return label_vader

# Chunked unique extraction for filter options (memory-safe)
def unique_values_from_csv(path_or_buffer, colname, chunksize=200_000):
    vals = set()
    try:
        if hasattr(path_or_buffer, "read"):
            # file-like (uploaded); we will attempt chunk reading, but ensure pointer reset
            try:
                path_or_buffer.seek(0)
            except Exception:
                pass
            for chunk in pd.read_csv(path_or_buffer, usecols=[colname], chunksize=chunksize, low_memory=False):
                vals.update(chunk[colname].dropna().unique().tolist())
            try:
                path_or_buffer.seek(0)
            except Exception:
                pass
        else:
            for chunk in pd.read_csv(path_or_buffer, usecols=[colname], chunksize=chunksize, low_memory=False):
                vals.update(chunk[colname].dropna().unique().tolist())
    except Exception as e:
        print(f"[unique_values_from_csv] failed for column {colname}: {e}")
    return sorted(v for v in vals if pd.notna(v))

def numeric_min_max_from_csv(path_or_buffer, colname, chunksize=200_000):
    """Return (min, max) for numeric column from CSV in chunks. Returns (None, None) if none."""
    mins = []
    maxs = []
    try:
        if hasattr(path_or_buffer, "read"):
            try:
                path_or_buffer.seek(0)
            except Exception:
                pass
            for chunk in pd.read_csv(path_or_buffer, usecols=[colname], chunksize=chunksize, low_memory=False):
                ser = pd.to_numeric(chunk[colname], errors="coerce").dropna()
                if not ser.empty:
                    mins.append(ser.min()); maxs.append(ser.max())
            try:
                path_or_buffer.seek(0)
            except Exception:
                pass
        else:
            for chunk in pd.read_csv(path_or_buffer, usecols=[colname], chunksize=chunksize, low_memory=False):
                ser = pd.to_numeric(chunk[colname], errors="coerce").dropna()
                if not ser.empty:
                    mins.append(ser.min()); maxs.append(ser.max())
    except Exception as e:
        print(f"[numeric_min_max_from_csv] failed for {colname}: {e}")
    if not mins:
        return None, None
    return float(min(mins)), float(max(maxs))

# --- UI ---

st.set_page_config(page_title="💄✨ Hybrid Sentiment Analysis", layout="wide")
st.title("💄✨ Hybrid Sentiment Analysis — Fashion & Cosmetics")

# 📘 User Manual (collapsible)
with st.expander("📘 User Manual", expanded=False):
    st.markdown("""
    **Quick guide**
    - Upload a CSV or pick a processed file from the sidebar.
    - Expected columns (recommended): `review_text` (or `text`), `review_rating`, `brand_name`, `product_id`, `product_title`, `review_date`, `price`, `verified_purchases`.
    - Use **Fast** mode on Cloud for a quick demo (samples), **Full** mode locally for full dataset processing.
    - If a filter returns no rows, try removing filters or switch to Full mode locally.
    """)

st.sidebar.header("Data & Controls")

# Upload or choose dataset
upload = st.sidebar.file_uploader("Upload a CSV file (optional)", type="csv")
available_files = list(Path(st.session_state["PROCESSED_DIR"]).glob("*.csv"))
# Show readable names but keep path for loading
sel_file = st.sidebar.selectbox("Choose processed dataset", available_files)

# Build filter option values (memory-safe) from full file (or uploaded file-like)
# We extract only necessary columns for filters: brand_name, price, verified_purchases
brand_options = []
price_min_max = (None, None)
verified_options_exist = False
try:
    if upload:
        brand_options = unique_values_from_csv(upload, "brand_name")
        price_min_max = numeric_min_max_from_csv(upload, "price")
        # detect verified column presence
        verified_options = unique_values_from_csv(upload, "verified_purchases")
        verified_options_exist = len(verified_options) > 0
    else:
        brand_options = unique_values_from_csv(sel_file, "brand_name")
        price_min_max = numeric_min_max_from_csv(sel_file, "price")
        verified_options = unique_values_from_csv(sel_file, "verified_purchases")
        verified_options_exist = len(verified_options) > 0
except Exception as e:
    print("[filter-prep] warning:", e)

# Load sample/safe dataset for analytics (not the full heavy read)
if upload:
    df = safe_read_csv(upload)
else:
    df = safe_read_csv(sel_file)

if df is None or df.empty:
    st.error("No data loaded. Upload a CSV or add processed CSVs to data/processed.")
    st.stop()

# normalize display column
display_col = "review_text" if "review_text" in df.columns else ("text" if "text" in df.columns else None)
if display_col is None:
    st.error("Dataset needs a text column named 'review_text' or 'text'.")
    st.stop()
df['text_clean'] = df[display_col].astype(str).fillna("")

# Fast vs Full: compute labels BEFORE filters (but in Fast mode compute only on a sample)
st.sidebar.markdown("### ⚡ Performance mode")
mode = st.sidebar.radio("Choose analysis mode:", ["Fast (sample only)", "Full (all rows)"], index=0)

if "hybrid_label" not in df.columns:
    if mode == "Full (all rows)":
        st.info("Computing hybrid labels for **all rows** — this may take a while (use locally).")
        df['hybrid_label'] = df.apply(hybrid_label, axis=1)
    else:
        st.warning("Hybrid labels not precomputed — using 1000-row sample for demo.")
        sample_df = safe_sample(df, 1000)
        sample_df['hybrid_label'] = sample_df.apply(hybrid_label, axis=1)
        df = sample_df.reset_index(drop=True)

label_col = "hybrid_label"

# --- Filters (applied after labels) ---
# Brand filter (use values from full CSV if available)
if brand_options:
    sel_brand = st.sidebar.multiselect("Filter by brand", brand_options)
else:
    sel_brand = []
    st.sidebar.info("No brand options to select. Upload a dataset or switch to Full mode.")

if sel_brand:
    # If df doesn't contain brand_name column (e.g. small demo), skip gracefully
    if 'brand_name' in df.columns:
        df = df[df['brand_name'].isin(sel_brand)]
    else:
        st.warning("Selected brand(s) exist in full file but not in the active sample. Try switching to Full mode.")

# Price filter (safe)
if price_min_max and price_min_max != (None, None):
    min_p, max_p = price_min_max
    # ensure sane order
    if min_p is not None and max_p is not None and min_p <= max_p:
        sel_p = st.sidebar.slider("Price range", min_p, max_p, (min_p, max_p))
        if 'price' in df.columns:
            df = df[(pd.to_numeric(df['price'], errors='coerce') >= sel_p[0]) & (pd.to_numeric(df['price'], errors='coerce') <= sel_p[1])]
    else:
        st.sidebar.info("Price values not available.")
else:
    st.sidebar.info("⚠️ Price data not available for this dataset")

# Verified purchases filter
if verified_options_exist:
    vp_filter = st.sidebar.radio("Verified purchase filter", ["All", "Verified only"])
    if vp_filter == "Verified only":
        if 'verified_purchases' in df.columns:
            # coerce to boolean-ish
            df = df[df['verified_purchases'].astype(str).str.lower().isin(['true','1','yes'])]
        else:
            st.sidebar.info("Verified purchase column not present in active sample; try Full mode.")
else:
    st.sidebar.info("No verified purchase info available in the dataset")

# After filters applied -> check emptiness
if df is None or df.empty:
    st.warning("No data after applying filters. Try removing filters or switch to Full mode for a complete run.")
    st.stop()

# --- Analytics ---

st.subheader("Analytics Dashboard (filtered)")
st.write(f"Rows in active view: {len(df)}")

# Label counts (safe)
st.write("Label counts:", df[label_col].value_counts().to_dict())

# Sentiment distribution (bar chart)
st.bar_chart(df[label_col].value_counts())

# WordClouds by sentiment
st.subheader("Top keywords by sentiment (filtered sample)")
sample_n = st.slider("Sample size for keywords", 100, 5000, 500)
sample_df = safe_sample(df[['text_clean', label_col]].dropna(), sample_n)

if sample_df is None or sample_df.empty:
    st.info("No text samples available for keyword extraction.")
else:
    for sentiment in ["positive", "negative", "neutral"]:
        text = " ".join(sample_df[sample_df[label_col]==sentiment]['text_clean'].astype(str).tolist())
        if text.strip():
            wc = WordCloud(width=600, height=400, background_color="white").generate(text)
            st.image(wc.to_array(), caption=f"WordCloud — {sentiment}")

# Product-level rollups
if 'product_id' in df.columns:
    st.subheader("Product-level rollups (filtered)")
    rollup = df.groupby(['product_id','product_title','brand_name'], dropna=False).agg(
        n_reviews = (label_col,'count'),
        positive_share = (label_col, lambda s: (s=="positive").mean()),
        avg_rating = ('review_rating','mean') if 'review_rating' in df.columns else ('rating','mean')
    ).reset_index()
    st.dataframe(rollup.head(50))
    csv = rollup.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download product rollups CSV", csv, "product_rollups.csv", "text/csv")
else:
    st.info("product_id column not present — product rollups unavailable.")

# Time trends
if 'review_date' in df.columns:
    st.subheader("Sentiment trends over time")
    time_df = df.copy()
    time_df['review_date'] = pd.to_datetime(time_df['review_date'], errors='coerce')
    time_df['_period'] = time_df['review_date'].dt.to_period('M')
    monthly = time_df.groupby('_period').agg(
        n_reviews = (label_col,'count'),
        positive_share = (label_col, lambda s: (s=="positive").mean()),
        avg_rating = ('review_rating','mean') if 'review_rating' in time_df.columns else ('rating','mean')
    ).reset_index()
    if not monthly.empty:
        st.line_chart(monthly.set_index('_period')[['positive_share','avg_rating']])
    else:
        st.info("Not enough dated reviews to show trends.")
else:
    st.info("review_date column not present — trends unavailable.")

# Single review prediction
st.subheader("Single review prediction")
user_text = st.text_area("Paste review text", "Love this lipstick — color pops and lasts!")
user_rating = st.number_input("Rating (if known)", min_value=1, max_value=5, step=1)

if st.button("Predict sentiment"):
    row = {"text_clean": user_text, "review_rating": user_rating}
    label = hybrid_label(row)
    st.success(f"Predicted sentiment: {label}")
