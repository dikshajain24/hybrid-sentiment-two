# app.py — Hybrid Sentiment Analysis Dashboard
# Fast-mode: sample first -> label -> filter (safe for Streamlit Cloud)
# Full-mode: label the active dataframe (use locally for very large datasets)

# --- STARTUP GUARD ---
import os
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ensure src package exists
if not (ROOT / "src" / "__init__.py").exists():
    (ROOT / "src").mkdir(parents=True, exist_ok=True)
    (ROOT / "src" / "__init__.py").write_text("# auto-created\n")

# create processed dir and a small demo csv if empty
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
    print("[startup] demo CSV created at", demo_csv)

# model loader (optional)
PIPE = None
MODEL_PATH = ROOT / "models" / "tfidf_lr.joblib"
try:
    import joblib
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        PIPE = joblib.load(MODEL_PATH)
        print("[startup] Model loaded")
    else:
        print("[startup] Model not found; continuing without ML model")
except Exception as e:
    print("[startup] Model load failed:", e)

st.session_state.setdefault("PIPE", PIPE)
st.session_state.setdefault("PROCESSED_DIR", str(processed_dir))

# Ensure vader lexicon present
try:
    import nltk
    nltk.data.find("sentiment/vader_lexicon.zip")
except Exception:
    try:
        nltk.download("vader_lexicon", quiet=True)
    except Exception as e:
        print("[startup] Failed to download vader lexicon:", e)
# --- END STARTUP GUARD ---

# imports
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------------- Helpers ----------------

def safe_read_csv(path_or_buffer, max_rows_for_ui=250000):
    """Read up to max_rows_for_ui rows for the interactive UI to avoid OOM/timeouts."""
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
            # filepath (read only first max_rows_for_ui rows)
            return pd.read_csv(path_or_buffer, nrows=max_rows_for_ui, low_memory=False)
    except Exception as e:
        print("[safe_read_csv] Failed:", e)
        return pd.DataFrame()

def safe_sample(df, n, random_state=42):
    """Return a sample of df or empty df if none."""
    if df is None or df.empty:
        return df.head(0)
    n_eff = min(max(0, int(n)), len(df))
    return df.sample(n=n_eff, random_state=random_state).reset_index(drop=True) if n_eff > 0 else df.head(0)

def ml_predict_safe(texts):
    """Return (preds, confs). If no model available returns (None, 0.0)."""
    pipe = st.session_state.get("PIPE", None)
    if pipe is None:
        return [None] * len(texts), [0.0] * len(texts)
    try:
        preds = pipe.predict(texts)
        try:
            probs = pipe.predict_proba(texts)
            confs = list(probs.max(axis=1))
        except Exception:
            confs = [0.0] * len(texts)
        return preds, confs
    except Exception as e:
        print("[ml_predict_safe] failed:", e)
        return [None] * len(texts), [0.0] * len(texts)

def hybrid_label(row):
    """Return a hybrid label (priority: confident ML -> rating heuristic -> vader)."""
    vader = SentimentIntensityAnalyzer()
    text = (row.get("text_clean") or "")
    score = vader.polarity_scores(text)["compound"]
    label_vader = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"

    preds, confs = ml_predict_safe([text])
    label_ml, conf = preds[0], confs[0]

    try:
        rating = float(row.get("review_rating", np.nan))
    except Exception:
        rating = np.nan

    if label_ml is not None and conf >= 0.8:
        return label_ml
    if not np.isnan(rating) and rating >= 4 and label_vader != "negative":
        return "positive"
    return label_vader

def unique_values_from_csv(path_or_buffer, colname, chunksize=200_000):
    """Return sorted unique non-null values for colname from CSV path or uploaded file-like (memory-safe)."""
    vals = set()
    try:
        if hasattr(path_or_buffer, "read"):
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
    """Return (min, max) numeric values for colname from CSV in chunks; (None, None) if missing."""
    mins, maxs = [], []
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

# ---------------- UI ----------------

st.set_page_config(page_title="💄✨ Hybrid Sentiment Analysis", layout="wide")
st.title("💄✨ Hybrid Sentiment Analysis — Fashion & Cosmetics")

# User manual as expander
with st.expander("📘 User Manual", expanded=False):
    st.markdown("""
    **Quick guide**
    - Upload a CSV or choose a processed file from the sidebar.
    - Recommended columns: `review_text` (or `text`), `review_rating`, `brand_name`, `product_id`, `product_title`, `review_date`, `price`, `verified_purchases`.
    - **Fast (sample)** mode: quick demo (1000 rows) suitable for Streamlit Cloud.
    - **Full (all rows)** mode: compute labels for whole active DataFrame — use locally for large datasets.
    - If filters return no rows, remove filters or switch to Full mode locally.
    """)

st.sidebar.header("Data & Controls")

# file upload or choose processed dataset
upload = st.sidebar.file_uploader("Upload a CSV file (optional)", type="csv")
available_files = list(Path(st.session_state["PROCESSED_DIR"]).glob("*.csv"))
sel_file = st.sidebar.selectbox("Choose processed dataset", available_files)

# Pre-scan necessary columns for filters from the full file (memory-safe)
brand_options = []
price_min_max = (None, None)
verified_options_exist = False

try:
    if upload:
        brand_options = unique_values_from_csv(upload, "brand_name")
        price_min_max = numeric_min_max_from_csv(upload, "price")
        verified_values = unique_values_from_csv(upload, "verified_purchases")
        verified_options_exist = len(verified_values) > 0
    else:
        brand_options = unique_values_from_csv(sel_file, "brand_name")
        price_min_max = numeric_min_max_from_csv(sel_file, "price")
        verified_values = unique_values_from_csv(sel_file, "verified_purchases")
        verified_options_exist = len(verified_values) > 0
except Exception as e:
    print("[filter-prep] warning:", e)

# Load an active dataframe for analytics (bounded size for UI)
if upload:
    df_active = safe_read_csv(upload)
else:
    df_active = safe_read_csv(sel_file)

if df_active is None or df_active.empty:
    st.error("No data loaded. Upload a CSV or add processed CSVs to data/processed.")
    st.stop()

# choose display text column
display_col = "review_text" if "review_text" in df_active.columns else ("text" if "text" in df_active.columns else None)
if display_col is None:
    st.error("Dataset must have 'review_text' or 'text' column.")
    st.stop()
df_active["text_clean"] = df_active[display_col].astype(str).fillna("")

# performance mode: we will handle Fast vs Full differently (Fast: sample FIRST -> label -> filters)
st.sidebar.markdown("### ⚡ Performance mode")
mode = st.sidebar.radio("Choose analysis mode:", ["Fast (sample only)", "Full (all rows)"], index=0)

# --- Sample-first for Fast --- compute labels on the sample, not on entire loaded df_active
if "hybrid_label" not in df_active.columns:
    if mode == "Full (all rows)":
        st.info("Computing hybrid labels for active dataframe (Full mode). This may take a while — run locally for big data.")
        # compute on full (active) dataframe (note: df_active is already bounded by safe_read_csv)
        df_active["hybrid_label"] = df_active.apply(hybrid_label, axis=1)
    else:
        st.warning("Hybrid labels not precomputed — sampling 1000 rows for a fast demo.")
        sample_df = safe_sample(df_active, 1000)
        sample_df["hybrid_label"] = sample_df.apply(hybrid_label, axis=1)
        df_active = sample_df

label_col = "hybrid_label"

# --- BUILD FILTER WIDGETS (use pre-scanned brand_options, price_min_max, verified_options_exist) ---
if brand_options:
    sel_brand = st.sidebar.multiselect("Filter by brand", brand_options)
else:
    sel_brand = []
    st.sidebar.info("No brand options to select. Upload a dataset or switch to Full mode.")

# Price slider using pre-scanned min/max
if price_min_max and price_min_max != (None, None):
    pmin, pmax = price_min_max
    if pmin is not None and pmax is not None and pmin <= pmax:
        sel_p = st.sidebar.slider("Price range", float(pmin), float(pmax), (float(pmin), float(pmax)))
    else:
        sel_p = None
        st.sidebar.info("Price values not available.")
else:
    sel_p = None
    st.sidebar.info("⚠️ Price data not available for this dataset")

# Verified purchases filter
if verified_options_exist:
    vp_filter = st.sidebar.radio("Verified purchase filter", ["All", "Verified only"])
else:
    vp_filter = "All"
    st.sidebar.info("No verified purchase info available in the dataset")

# --- APPLY FILTERS (after labels exist) ---
# brand
if sel_brand:
    if "brand_name" in df_active.columns:
        df_active = df_active[df_active["brand_name"].isin(sel_brand)]
    else:
        st.warning("Selected brand(s) exist in the full file but not in the active sample. Try Full mode for a complete run.")

# price
if sel_p is not None and "price" in df_active.columns:
    df_active = df_active[(pd.to_numeric(df_active["price"], errors="coerce") >= sel_p[0]) &
                          (pd.to_numeric(df_active["price"], errors="coerce") <= sel_p[1])]

# verified purchases
if vp_filter == "Verified only":
    if "verified_purchases" in df_active.columns:
        df_active = df_active[df_active["verified_purchases"].astype(str).str.lower().isin(["true","1","yes"])]
    else:
        st.sidebar.info("Verified column not present in active sample; try Full mode.")

# if filters removed everything, stop gracefully
if df_active is None or df_active.empty:
    st.warning("No data after applying filters. Try removing filters or switch to Full mode for a complete run.")
    st.stop()

# ---------------- Analytics (based on df_active) ----------------

st.subheader("Analytics Dashboard (filtered)")
st.write(f"Rows in active view: {len(df_active)}")

# label counts
st.write("Label counts:", df_active[label_col].value_counts().to_dict())

# sentiment distribution
st.bar_chart(df_active[label_col].value_counts())

# Top keywords by sentiment (wordclouds)
st.subheader("Top keywords by sentiment (filtered sample)")
sample_n = st.slider("Sample size for keywords", 100, 5000, 500)
kw_sample = safe_sample(df_active[["text_clean", label_col]].dropna(), sample_n)
if kw_sample is None or kw_sample.empty:
    st.info("No text samples available for keyword extraction.")
else:
    for sentiment in ["positive", "negative", "neutral"]:
        text = " ".join(kw_sample[kw_sample[label_col] == sentiment]["text_clean"].astype(str).tolist())
        if text.strip():
            wc = WordCloud(width=600, height=400, background_color="white").generate(text)
            st.image(wc.to_array(), caption=f"WordCloud — {sentiment}")

# Product-level rollups & download
if "product_id" in df_active.columns:
    st.subheader("Product-level rollups (filtered)")
    rollup = df_active.groupby(["product_id", "product_title", "brand_name"], dropna=False).agg(
        n_reviews = (label_col, "count"),
        positive_share = (label_col, lambda s: (s == "positive").mean()),
        avg_rating = ("review_rating", "mean") if "review_rating" in df_active.columns else ("rating", "mean")
    ).reset_index()
    st.dataframe(rollup.head(50))
    csv_out = rollup.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download product rollups CSV", csv_out, "product_rollups.csv", "text/csv")
else:
    st.info("product_id column not present — product rollups unavailable.")

# Time trends
if "review_date" in df_active.columns:
    st.subheader("Sentiment trends over time")
    time_df = df_active.copy()
    time_df["review_date"] = pd.to_datetime(time_df["review_date"], errors="coerce")
    time_df["_period"] = time_df["review_date"].dt.to_period("M")
    monthly = time_df.groupby("_period").agg(
        n_reviews = (label_col, "count"),
        positive_share = (label_col, lambda s: (s == "positive").mean()),
        avg_rating = ("review_rating", "mean") if "review_rating" in time_df.columns else ("rating", "mean")
    ).reset_index()
    if not monthly.empty:
        st.line_chart(monthly.set_index("_period")[["positive_share", "avg_rating"]])
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

# End of file
