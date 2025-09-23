# app.py — Hybrid Sentiment Analysis Dashboard (with Fast/Full toggle + User Manual)

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
        "review_id,product_id,brand_name,review_text,review_rating,review_date\n"
        "1,1001,DemoBrand,\"Love this lipstick — color pops and lasts!\",5,2023-08-01\n"
        "2,1001,DemoBrand,\"Not what I expected, too dry\",2,2023-07-15\n"
        "3,1002,DemoBrand,\"Okay for the price\",3,2023-06-10\n"
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
    try:
        if isinstance(path_or_buffer, (str, Path)):
            return pd.read_csv(path_or_buffer, nrows=max_rows_for_ui, low_memory=False)
        else:
            df = pd.read_csv(path_or_buffer, low_memory=False)
            if len(df) > max_rows_for_ui:
                return df.sample(n=max_rows_for_ui, random_state=42).reset_index(drop=True)
            return df
    except Exception as e:
        print("[safe_read_csv] Failed:", e)
        return pd.DataFrame()

def safe_sample(df, n, random_state=42):
    if df is None or df.empty:
        return df.head(0)
    n_eff = min(max(0, int(n)), len(df))
    return df.sample(n=n_eff, random_state=random_state) if n_eff > 0 else df.head(0)

def ml_predict_safe(texts):
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
    score = vader.polarity_scores(row['text_clean'])['compound']
    label_vader = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"

    preds, confs = ml_predict_safe([row['text_clean']])
    label_ml, conf = preds[0], confs[0]

    if label_ml is not None and conf >= 0.8:
        return label_ml
    if row.get('review_rating', np.nan) >= 4 and label_vader != "negative":
        return "positive"
    return label_vader

# --- UI ---

st.set_page_config(page_title="💄✨ Hybrid Sentiment Analysis", layout="wide")
st.title("💄✨ Hybrid Sentiment Analysis — Fashion & Cosmetics")

# 📘 User Manual / Instructions
st.markdown("""
## 📘 User Manual
Welcome to the **Hybrid Sentiment Analysis Dashboard** for Fashion & Cosmetics!  

Here’s how to use it:
1. **Upload a CSV** or choose from preprocessed datasets (sidebar).  
   - Expected columns: `review_text` (or `text`), `review_rating`, `brand_name`, `product_id`, `review_date`.  
2. **Filters**: Narrow down reviews by brand, price range, or verified purchase.  
3. **Dashboard Sections**:
   - **Sentiment Distribution** → See how reviews split across positive/negative/neutral.  
   - **Word Clouds** → Top keywords per sentiment.  
   - **Product Rollups** → Review counts, avg rating, positive share per product.  
   - **Trends** → Sentiment and rating trends over time.  
   - **Download CSV** → Export aggregated product insights.  
   - **Single Review Prediction** → Paste your own review and get instant sentiment.  
4. **Performance Mode**:
   - *Fast (sample only)* → Loads quickly (default on Cloud).  
   - *Full (all rows)* → Computes sentiment for the entire dataset (use locally).  

⚡ Tip: On **Streamlit Cloud**, always use **Fast Mode** for responsiveness.
""")

st.sidebar.header("Data & Controls")

# Upload or choose dataset
upload = st.sidebar.file_uploader("Upload a CSV file", type="csv")
available_files = list(Path(st.session_state["PROCESSED_DIR"]).glob("*.csv"))
sel_file = st.sidebar.selectbox("Choose processed dataset", available_files)

if upload:
    df = safe_read_csv(upload)
else:
    df = safe_read_csv(sel_file)

if df.empty:
    st.error("No data loaded. Please upload a CSV or check processed folder.")
    st.stop()

# Pick display + label columns
display_col = "review_text" if "review_text" in df.columns else "text"
df['text_clean'] = df[display_col].astype(str).fillna("")

# 🚀 Fast vs Full toggle
st.sidebar.markdown("### ⚡ Performance mode")
mode = st.sidebar.radio("Choose analysis mode:", ["Fast (sample only)", "Full (all rows)"], index=0)

if "hybrid_label" not in df.columns:
    if mode == "Full (all rows)":
        st.info("Computing hybrid labels for **all rows** — may take a while ⏳")
        df['hybrid_label'] = df.apply(hybrid_label, axis=1)
    else:
        st.warning("Hybrid labels not precomputed — using 1000-row sample for demo.")
        sample_df = safe_sample(df, 1000)
        sample_df['hybrid_label'] = sample_df.apply(hybrid_label, axis=1)
        df = sample_df.reset_index(drop=True)

label_col = "hybrid_label"

# Filters
brands = df['brand_name'].dropna().unique().tolist() if 'brand_name' in df.columns else []
sel_brand = st.sidebar.multiselect("Filter by brand", brands)
if sel_brand:
    df = df[df['brand_name'].isin(sel_brand)]

if 'price' in df.columns:
    min_p, max_p = float(df['price'].min()), float(df['price'].max())
    sel_p = st.sidebar.slider("Price range", min_p, max_p, (min_p, max_p))
    df = df[(df['price'] >= sel_p[0]) & (df['price'] <= sel_p[1])]

if 'verified_purchases' in df.columns:
    vp_filter = st.sidebar.radio("Verified purchase filter", ["All", "Verified only"])
    if vp_filter == "Verified only":
        df = df[df['verified_purchases'] == True]

# --- Analytics ---

st.subheader("Analytics Dashboard (filtered)")
st.write("Rows:", len(df))

st.write("Label counts:", df[label_col].value_counts().to_dict())

# Sentiment distribution
st.bar_chart(df[label_col].value_counts())

# WordClouds by sentiment
st.subheader("Top keywords by sentiment (filtered sample)")
sample_n = st.slider("Sample size for keywords", 100, 5000, 500)
sample_df = safe_sample(df[['text_clean', label_col]].dropna(), sample_n)

if not sample_df.empty:
    for sentiment in ["positive", "negative", "neutral"]:
        text = " ".join(sample_df[sample_df[label_col]==sentiment]['text_clean'])
        if text:
            wc = WordCloud(width=600, height=400, background_color="white").generate(text)
            st.image(wc.to_array(), caption=f"WordCloud — {sentiment}")

# Product-level rollups
if 'product_id' in df.columns:
    st.subheader("Product-level rollups")
    rollup = df.groupby(['product_id','product_title','brand_name']).agg(
        n_reviews = (label_col,'count'),
        positive_share = (label_col, lambda s: (s=="positive").mean()),
        avg_rating = ('review_rating','mean') if 'review_rating' in df.columns else ('rating','mean')
    ).reset_index()
    st.dataframe(rollup.head(20))
    csv = rollup.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download product rollups CSV", csv, "product_rollups.csv", "text/csv")

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
    st.line_chart(monthly.set_index('_period')[['positive_share','avg_rating']])

# Single review prediction
st.subheader("Single review prediction")
user_text = st.text_area("Paste review text", "Love this lipstick — color pops and lasts!")
user_rating = st.number_input("Rating (if known)", min_value=1, max_value=5, step=1)

if st.button("Predict sentiment"):
    row = {"text_clean": user_text, "review_rating": user_rating}
    label = hybrid_label(row)
    st.success(f"Predicted sentiment: {label}")
