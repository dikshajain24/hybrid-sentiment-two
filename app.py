# app.py
import streamlit as st
import joblib
from pathlib import Path
from src import utils
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import io

# Initialize NLTK/VADER
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

# ---------------------------
# Page setup & styling
# ---------------------------
st.set_page_config(page_title="Hybrid Sentiment — Fashion & Cosmetics", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg,#f7e8f3,#f0f1ff,#fdefec); font-family: 'Poppins', sans-serif; color: #222; }
    h1,h2,h3{ text-align:center; color:#6a1b9a; }
    .card{ background:white; padding:14px; border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,0.08); }
    div.stButton>button{ background-color:#ab47bc; color:white; border-radius:12px; padding:8px 16px; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True
)

sia = SentimentIntensityAnalyzer()

# ---------------------------
# Model loader (cached)
# ---------------------------
@st.cache_resource
def load_model():
    model_path = Path(utils.ROOT) / "models" / "tfidf_lr.joblib"
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None

pipe = load_model()

# ---------------------------
# Header
# ---------------------------
st.title("💄✨ Hybrid Sentiment Analysis — Fashion & Cosmetics")
st.markdown("Upload a CSV or choose a processed dataset. Explore sentiment distribution, keywords, product rollups, and time trends.")

# ---------------------------
# Processed files list
# ---------------------------
processed_dir = utils.processed_dir()
processed_dir.mkdir(parents=True, exist_ok=True)
combined_files = list(processed_dir.glob("combined_*.csv")) + list(processed_dir.glob("*.csv"))
combined_map = {p.name: p for p in combined_files}

# ---------------------------
# Left column: uploader / file selection / filters / single predict
# ---------------------------
left, right = st.columns([1, 2])
with left:
    st.header("Data & Controls")

    # --- Manual / Help button & panel (place inside left column) ---
    manual_text = """
### Manual — Quick Guide

**What this app does**
- Runs hybrid sentiment (lexicon + ML) on product reviews.
- Shows distribution, keywords, product rollups, monthly trends.
- Upload a CSV or choose a processed file from `data/processed/`.

**Quick steps**
1. Upload a CSV or choose a processed dataset from the dropdown.  
2. Use filters (Brand / Price / Verified) to narrow the data.  
3. View charts on the right: distribution, keywords, examples, product rollups, time trends.  
4. Export results: download filtered CSV or top-N products.  
5. Use *Single review prediction* to paste one review and get an instant label.

**CSV format (recommended)**
- Required: `review_text` or `text` (the app auto-detects common text columns).  
- Optional but useful: `product_id`, `product_title`, `brand_name`, `review_rating` (or `rating`), `review_date` (or `timestamp`), `verified_purchases` (or `is_a_buyer`).

**Label rules (simple)**
- If ML model exists and `ml_conf >= 0.80` → use ML label.  
- Else if rating >= 4 and lexicon (VADER) not negative → positive.  
- Else → VADER label.

**Tips**
- Upload a small sample (100–500 rows) to test quickly.  
- For big files, precompute offline with `python -m src.hybrid`.  
- If examples are empty, clear filters or check the debug label counts printed near the top.
"""

    if 'show_manual' not in st.session_state:
        st.session_state['show_manual'] = False

    if st.button("❓ Manual / Help"):
        st.session_state['show_manual'] = not st.session_state['show_manual']

    with st.expander("Manual — click to open", expanded=st.session_state['show_manual']):
        st.markdown(manual_text)

    # -------------------------
    upload = st.file_uploader("Upload a CSV file (optional)", type=["csv"],
                              help="Upload raw or processed CSV. App will analyze it in-session. Use 'Save processed' to store it locally.")
    use_uploaded = upload is not None

    if not use_uploaded:
        if combined_map:
            sel_file = st.selectbox("Choose processed dataset", options=list(combined_map.keys()))
            df = pd.read_csv(combined_map[sel_file], low_memory=False)
        else:
            st.warning("No processed CSVs found in data/processed. Upload a file or run the pipeline to create processed files.")
            df = pd.DataFrame()
    else:
        # read uploaded file
        try:
            df = pd.read_csv(upload, low_memory=False)
        except Exception:
            df = pd.read_csv(upload, encoding="utf-8-sig", low_memory=False)
        st.success(f"Uploaded file with {len(df):,} rows.")
        if len(df) > 250_000:
            st.warning("Large file detected — analysis may be slow. Consider uploading a smaller sample (<250k rows).")

    # Ensure a text_clean column exists (prefer common columns)
    if not df.empty:
        if 'text_clean' not in df.columns:
            if 'review_text' in df.columns:
                df['text_clean'] = df['review_text'].fillna("").astype(str).apply(utils.clean_text)
            elif 'text' in df.columns:
                df['text_clean'] = df['text'].fillna("").astype(str).apply(utils.clean_text)
            else:
                # try auto-detect
                text_cols = [c for c in df.columns if any(k in c.lower() for k in ('review','text','comment','body'))]
                if text_cols:
                    df['text_clean'] = df[text_cols[0]].fillna("").astype(str).apply(utils.clean_text)
                else:
                    df['text_clean'] = df.astype(str).agg(' '.join, axis=1).apply(utils.clean_text)

    st.markdown(f"**Rows:** {len(df):,} — columns: {', '.join(df.columns[:8])} ...")

    st.markdown("---")
    st.subheader("Interactive filters")
    # Brand filter
    brands = sorted(df['brand_name'].dropna().unique().astype(str)) if 'brand_name' in df.columns else []
    chosen_brands = st.multiselect("Filter by brand", options=brands, default=None)

    # Price filter (price or mrp)
    price_col = 'price' if 'price' in df.columns else ('mrp' if 'mrp' in df.columns else None)
    if price_col:
        df['_price_num'] = pd.to_numeric(df[price_col], errors='coerce')
        pmin = float(df['_price_num'].min(skipna=True)) if not df['_price_num'].isna().all() else 0.0
        pmax = float(df['_price_num'].max(skipna=True)) if not df['_price_num'].isna().all() else 100.0
        low, high = st.slider("Price range", float(pmin), float(pmax), (float(pmin), float(pmax)))
    else:
        low, high = None, None

    # Verified purchase filter
    verified_col = 'verified_purchases' if 'verified_purchases' in df.columns else ('is_a_buyer' if 'is_a_buyer' in df.columns else None)
    verified_choice = None
    if verified_col:
        verified_choice = st.selectbox("Verified purchase filter", options=["All", "Only verified", "Only unverified"], index=0)

    st.markdown("---")
    st.subheader("Single review prediction")
    review_text = st.text_area("Paste review text", height=140, placeholder="E.g., 'Love this lipstick — color pops and lasts!'")
    rating = st.slider("Rating (if known)", 0, 5, 0)
    predict_btn = st.button("🔮 Predict review")

    # Optionally save uploaded processed file
    if use_uploaded and not df.empty:
        if st.checkbox("Save uploaded dataset to data/processed for reuse?"):
            safe_name = st.text_input("Filename (no extension)", value="uploaded_processed")
            if st.button("Save now"):
                out_path = processed_dir / f"{safe_name}.csv"
                df.to_csv(out_path, index=False)
                st.success(f"Saved processed upload to {out_path.name}")

# ---------------------------
# Apply filters -> filtered_df
# ---------------------------
filtered_df = df.copy() if not df.empty else pd.DataFrame()
if not filtered_df.empty:
    if chosen_brands:
        filtered_df = filtered_df[filtered_df['brand_name'].isin(chosen_brands)].copy()
    if price_col and low is not None and high is not None:
        filtered_df = filtered_df[(filtered_df['_price_num'].notna()) & (filtered_df['_price_num'] >= low) & (filtered_df['_price_num'] <= high)].copy()
    if verified_col and verified_choice and verified_choice != "All":
        def is_verified_val(x):
            if pd.isna(x): return False
            if isinstance(x, (int, float)): return int(x) == 1
            if isinstance(x, bool): return x
            s = str(x).strip().lower()
            return s in ['true','1','yes','y','verified','t']
        if verified_choice == "Only verified":
            filtered_df = filtered_df[filtered_df[verified_col].apply(is_verified_val)].copy()
        else:
            filtered_df = filtered_df[~filtered_df[verified_col].apply(is_verified_val)].copy()

# ---------------------------
# Right column: analytics operate on filtered_df
# ---------------------------
with right:
    st.header("Analytics Dashboard (filtered)")

    if filtered_df.empty:
        st.info("No data to show. Upload a file or choose a processed dataset and adjust filters.")
        st.stop()

    # ---- Compute/ensure VADER scores ----
    if 'vader_compound' not in filtered_df.columns or 'vader_label' not in filtered_df.columns:
        with st.spinner("Computing VADER lexicon scores..."):
            filtered_df['vader_compound'] = filtered_df['text_clean'].fillna("").apply(lambda t: sia.polarity_scores(str(t))['compound'])
            filtered_df['vader_label'] = filtered_df['vader_compound'].apply(lambda s: 'positive' if s >= 0.05 else ('negative' if s <= -0.05 else 'neutral'))

    # ---- Compute/ensure ML preds if model available ----
    if pipe is not None:
        try:
            if 'ml_pred' not in filtered_df.columns or 'ml_conf' not in filtered_df.columns:
                with st.spinner("Applying ML model to filtered data..."):
                    X_text = filtered_df['text_clean'].fillna("").astype(str)
                    try:
                        preds = pipe.predict(X_text)
                        filtered_df['ml_pred'] = preds
                        try:
                            probs = pipe.predict_proba(X_text)
                            filtered_df['ml_conf'] = np.max(probs, axis=1)
                        except Exception:
                            filtered_df['ml_conf'] = 0.0
                    except Exception:
                        filtered_df['ml_pred'] = None
                        filtered_df['ml_conf'] = 0.0
        except Exception as e:
            st.warning(f"ML prediction failed on filtered set: {e}")

    # numeric ml_pred -> map to text if needed
    if 'ml_pred' in filtered_df.columns and filtered_df['ml_pred'].dtype.kind in 'iuf':
        uniq = sorted(filtered_df['ml_pred'].dropna().unique().tolist())
        mapping = {}
        if len(uniq) == 3:
            mapping = {uniq[0]: 'negative', uniq[1]: 'neutral', uniq[2]: 'positive'}
        elif len(uniq) == 2:
            mapping = {uniq[0]: 'negative', uniq[1]: 'positive'}
        elif len(uniq) == 1:
            mapping = {uniq[0]: 'positive'}
        if mapping:
            filtered_df['ml_pred'] = filtered_df['ml_pred'].map(mapping)

    if 'ml_pred' in filtered_df.columns:
        filtered_df['ml_pred'] = filtered_df['ml_pred'].fillna('').astype(str).str.lower().str.strip()

    # ---- rating_final column ----
    if 'rating_final' not in filtered_df.columns:
        if 'review_rating' in filtered_df.columns:
            filtered_df['rating_final'] = pd.to_numeric(filtered_df['review_rating'], errors='coerce')
        elif 'rating' in filtered_df.columns:
            filtered_df['rating_final'] = pd.to_numeric(filtered_df['rating'], errors='coerce')
        else:
            filtered_df['rating_final'] = np.nan

    # ---- Compute hybrid_label if missing ----
    if 'hybrid_label' not in filtered_df.columns:
        def hybrid_decision_row(r):
            vader = r.get('vader_label', 'neutral')
            mlp = r.get('ml_pred', None)
            mconf = float(r.get('ml_conf', 0.0)) if not pd.isna(r.get('ml_conf', None)) else 0.0
            rating_val = r.get('rating_final', np.nan)
            if mlp is not None and str(mlp) != 'nan' and mconf >= 0.80:
                return mlp
            if not pd.isna(rating_val) and rating_val >= 4:
                return 'positive' if str(vader) != 'negative' else 'neutral'
            return vader
        with st.spinner("Computing hybrid labels..."):
            filtered_df['hybrid_label'] = filtered_df.apply(hybrid_decision_row, axis=1)

    # ---- Normalize labels to canonical set and pick display column ----
    if 'hybrid_label' in filtered_df.columns:
        filtered_df['hybrid_label'] = filtered_df['hybrid_label'].fillna('').astype(str).str.lower().str.strip()
    if 'vader_label' in filtered_df.columns:
        filtered_df['vader_label'] = filtered_df['vader_label'].fillna('').astype(str).str.lower().str.strip()

    label_map = {
        'pos': 'positive', 'positive': 'positive', 'p': 'positive',
        'neg': 'negative', 'negative': 'negative', 'n': 'negative',
        'neu': 'neutral', 'neutral': 'neutral', 'ntr': 'neutral', '': 'neutral'
    }
    if 'hybrid_label' in filtered_df.columns:
        filtered_df['hybrid_label'] = filtered_df['hybrid_label'].map(label_map).fillna(filtered_df['hybrid_label'])
    if 'vader_label' in filtered_df.columns:
        filtered_df['vader_label'] = filtered_df['vader_label'].map(label_map).fillna(filtered_df['vader_label'])

    # Choose a display text column for examples
    if 'text_clean' in filtered_df.columns and filtered_df['text_clean'].notna().sum() > 0:
        DISPLAY_TEXT_COL = 'text_clean'
    elif 'review_text' in filtered_df.columns and filtered_df['review_text'].notna().sum() > 0:
        DISPLAY_TEXT_COL = 'review_text'
    elif 'text' in filtered_df.columns and filtered_df['text'].notna().sum() > 0:
        DISPLAY_TEXT_COL = 'text'
    else:
        filtered_df['__display_text'] = filtered_df.fillna('').astype(str).agg(' '.join, axis=1)
        DISPLAY_TEXT_COL = '__display_text'

    # DEBUG: counts to help you understand why everything might be neutral
    st.write(f"Using display column: `{DISPLAY_TEXT_COL}`")
    st.write("Label counts (hybrid):", filtered_df['hybrid_label'].value_counts().to_dict())
    if 'vader_label' in filtered_df.columns:
        st.write("Label counts (vader):", filtered_df['vader_label'].value_counts().to_dict())
    if 'ml_pred' in filtered_df.columns:
        st.write("Label counts (ml_pred):", filtered_df['ml_pred'].value_counts().to_dict())
        if 'ml_conf' in filtered_df.columns:
            st.write("ML confidence stats:", {
                "min": float(np.nanmin(filtered_df['ml_conf'])),
                "max": float(np.nanmax(filtered_df['ml_conf'])),
                "median": float(np.nanmedian(filtered_df['ml_conf']))
            })

    # --------------------------
    # Provide a relaxed-display option so you can inspect more ML predictions interactively
    # --------------------------
    st.markdown("---")
    st.subheader("Display options")
    relax_checkbox = st.checkbox("Use relaxed ML confidence threshold for display (don't change stored hybrid labels)", value=False)
    ml_conf_threshold = st.slider("ML confidence threshold for relaxed display", min_value=0.0, max_value=1.0, value=0.6, step=0.05) if relax_checkbox else 0.8

    # build display_label_col (applies relxation if chosen)
    def display_label_for_row(r, ml_conf_thresh=0.8):
        mlp = r.get('ml_pred', None)
        mconf = float(r.get('ml_conf', 0.0)) if not pd.isna(r.get('ml_conf', None)) else 0.0
        if mlp and mconf >= ml_conf_thresh:
            return mlp
        # rating override (same business rule)
        rating_val = r.get('rating_final', np.nan)
        vader = r.get('vader_label', 'neutral')
        if not pd.isna(rating_val) and rating_val >= 4 and str(vader) != 'negative':
            return 'positive'
        # fallback to hybrid if exists else vader
        if 'hybrid_label' in r.index and pd.notna(r['hybrid_label']):
            return r['hybrid_label']
        return vader

    # make a temporary column used only for display calculations
    filtered_df['_display_label'] = filtered_df.apply(lambda r: display_label_for_row(r, ml_conf_threshold), axis=1)
    display_label_col = '_display_label'

    # --- Sentiment distribution (using display_label_col) ---
    st.subheader("Sentiment distribution (filtered & display)")
    dist = filtered_df[display_label_col].fillna("unknown").value_counts().reindex(['positive','neutral','negative']).fillna(0)
    fig1, ax1 = plt.subplots()
    ax1.bar(dist.index.astype(str), dist.values, color=["#81c784","#fff176","#e57373"])
    ax1.set_title("Sentiment counts (display)")
    ax1.set_ylabel("Number of reviews")
    for i, v in enumerate(dist.values):
        ax1.text(i, v + max(dist.values)*0.01, str(int(v)), ha='center', va='bottom')
    st.pyplot(fig1)

    st.markdown("---")
    st.subheader("Top keywords by sentiment (filtered sample)")

    # Safe sample slider: min 1, max = max(1, min(5000, len(filtered_df)))
    max_sample = max(1, min(5000, len(filtered_df)))
    sample_n = st.slider("Sample size for keywords", min_value=1, max_value=max_sample, value=min(2000, max_sample), step=1)

    # Build a clean sample DF safely (avoid sample(n=0))
    sample_df_all = filtered_df[[DISPLAY_TEXT_COL, display_label_col]].dropna()
    if sample_df_all.empty:
        sample_df = pd.DataFrame(columns=[DISPLAY_TEXT_COL, display_label_col])
    else:
        sample_n_eff = min(sample_n, len(sample_df_all))
        sample_df = sample_df_all.sample(n=sample_n_eff, random_state=42)

    if not sample_df.empty:
        vect = CountVectorizer(stop_words='english', ngram_range=(1,2), max_features=4000)
        X = vect.fit_transform(sample_df[DISPLAY_TEXT_COL])
        feature_names = np.array(vect.get_feature_names_out())

        def top_terms_for_label(label, top_k=15):
            mask = sample_df[display_label_col] == label
            if mask.sum() == 0: 
                return []
            subX = X[mask.values]
            freqs = np.asarray(subX.sum(axis=0)).ravel()
            top_idx = freqs.argsort()[::-1][:top_k]
            return list(zip(feature_names[top_idx], freqs[top_idx]))

        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Positive — top terms**")
            pos_terms = top_terms_for_label('positive', 12)
            if pos_terms:
                st.table(pd.DataFrame(pos_terms, columns=['term','count']))
            else:
                st.write("—")
        with cols[1]:
            st.markdown("**Negative — top terms**")
            neg_terms = top_terms_for_label('negative', 12)
            if neg_terms:
                st.table(pd.DataFrame(neg_terms, columns=['term','count']))
            else:
                st.write("—")
    else:
        st.write("No text samples available for keyword extraction.")

    st.markdown("---")
    st.subheader("Review examples (filtered & display)")
    ex_cols = st.columns(3)
    for i, lab in enumerate(['positive','neutral','negative']):
        with ex_cols[i]:
            st.markdown(f"**{lab.title()} samples**")
            examples = filtered_df[filtered_df[display_label_col]==lab][DISPLAY_TEXT_COL].dropna().astype(str).head(6).tolist()
            if not examples:
                st.write("—")
            else:
                for ex in examples:
                    st.write("• " + (ex if len(ex) < 180 else ex[:177] + "…"))

    # -----------------------------
    # Download filtered processed data (key columns)
    # -----------------------------
    st.markdown("---")
    st.subheader("Download results (filtered)")
    export_cols = [
        'review_id','product_id','brand_name','product_title','review_text',
        'review_rating','rating','hybrid_label','vader_label','ml_pred','ml_conf'
    ]
    export_cols = [c for c in export_cols if c in filtered_df.columns]
    csv_bytes = filtered_df[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download filtered dataset with hybrid labels",
        data=csv_bytes,
        file_name="sentiment_results_filtered.csv",
        mime="text/csv"
    )

    # -----------------------------
    # Product-level rollups (filtered)
    # -----------------------------
    st.markdown("---")
    st.subheader("🛍️ Product-level sentiment summary (filtered)")
    if 'product_id' in filtered_df.columns:
        filtered_df['_is_positive'] = (filtered_df[display_label_col] == 'positive').astype(int)
        roll = filtered_df.groupby('product_id').agg(
            avg_rating=('rating_final','mean') if 'rating_final' in filtered_df.columns else ('review_rating','mean'),
            n_reviews=('review_id','count') if 'review_id' in filtered_df.columns else (DISPLAY_TEXT_COL,'count'),
            positive_share=('_is_positive','mean')
        ).reset_index()
        if 'brand_name' in filtered_df.columns:
            brand_lookup = filtered_df[['product_id','brand_name']].drop_duplicates(subset=['product_id'])
            roll = roll.merge(brand_lookup, on='product_id', how='left')
        if 'product_title' in filtered_df.columns:
            title_lookup = filtered_df[['product_id','product_title']].drop_duplicates(subset=['product_id'])
            roll = roll.merge(title_lookup, on='product_id', how='left')

        st.dataframe(roll.head(15))

        top_roll = roll.sort_values('positive_share', ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(7,4))
        ax2.barh(top_roll['product_title'].fillna(top_roll['product_id']).astype(str), top_roll['positive_share'])
        ax2.set_title("Top products by positive sentiment share")
        ax2.set_xlabel("Positive sentiment share")
        ax2.invert_yaxis()
        st.pyplot(fig2)

        # Export top-N
        st.markdown("---")
        st.subheader("Export top-N products")
        sort_choice = st.selectbox("Sort top products by", options=['positive_share','n_reviews','avg_rating'], index=0)
        top_n = st.number_input("Top N products to export", min_value=1, max_value=200, value=20, step=1)
        top_export = roll.sort_values(sort_choice, ascending=False).head(top_n)
        st.dataframe(top_export)
        buffer = io.StringIO()
        top_export.to_csv(buffer, index=False)
        st.download_button(
            label=f"📤 Download top-{top_n} products (by {sort_choice})",
            data=buffer.getvalue().encode("utf-8"),
            file_name=f"top_{top_n}_products_by_{sort_choice}.csv",
            mime="text/csv"
        )
    else:
        st.info("No `product_id` column found; product rollups unavailable.")

    # -----------------------------
    # Time-series sentiment trend (filtered)
    # -----------------------------
    st.markdown("---")
    st.subheader("📈 Sentiment over time (filtered)")
    date_col = 'review_date' if 'review_date' in filtered_df.columns else ('timestamp' if 'timestamp' in filtered_df.columns else None)
    if date_col:
        filtered_df['_parsed_date'] = pd.to_datetime(filtered_df[date_col], errors='coerce')
        time_df = filtered_df.dropna(subset=['_parsed_date']).copy()
        if not time_df.empty:
            time_df['_period'] = time_df['_parsed_date'].dt.to_period('M').dt.to_timestamp()
            monthly = time_df.groupby('_period').apply(lambda g: pd.Series({
                'n_reviews': len(g),
                'positive_share': (g[display_label_col]=='positive').mean(),
                'avg_rating': pd.to_numeric(g.get('review_rating', g.get('rating', np.nan))).mean()
            })).reset_index()
            fig3, ax3 = plt.subplots(figsize=(9,3))
            ax3.plot(monthly['_period'], monthly['positive_share'], marker='o')
            ax3.set_title("Monthly positive sentiment share (filtered)")
            ax3.set_ylabel("Positive share (0-1)")
            ax3.set_xlabel("Month")
            ax3.set_ylim(0,1)
            for x,y in zip(monthly['_period'], monthly['positive_share']):
                ax3.text(x, y + 0.01, f"{y:.2f}", ha='center', va='bottom', fontsize=8)
            st.pyplot(fig3)
            st.dataframe(monthly.tail(24).assign(_period=lambda d: d['_period'].dt.strftime('%Y-%m')).rename(columns={'_period':'month'}).reset_index(drop=True))
        else:
            st.info("No valid dates found after parsing.")
    else:
        st.info("No date column (`review_date` or `timestamp`) found for time-series analysis.")

# ---------------------------
# Single-review prediction
# ---------------------------
if predict_btn:
    if not str(review_text).strip():
        st.warning("Please enter review text to predict.")
    else:
        cleaned = utils.clean_text(review_text)
        vader = sia.polarity_scores(cleaned)['compound']
        vader_label = "positive" if vader >= 0.05 else ("negative" if vader <= -0.05 else "neutral")
        if pipe is not None:
            try:
                ml_pred = pipe.predict([cleaned])[0]
                try:
                    ml_conf = float(np.max(pipe.predict_proba([cleaned])))
                except Exception:
                    ml_conf = 0.0
            except Exception as e:
                ml_pred, ml_conf = "N/A", 0.0
                st.warning(f"Model prediction failed: {e}")
        else:
            ml_pred, ml_conf = "N/A", 0.0

        # numeric ml_pred mapping if needed
        if isinstance(ml_pred, (int, float, np.integer, np.floating)):
            if ml_pred == 2:
                ml_pred = 'positive'
            elif ml_pred == 1:
                ml_pred = 'neutral'
            elif ml_pred == 0:
                ml_pred = 'negative'
            else:
                ml_pred = str(ml_pred)

        if isinstance(ml_pred, str):
            ml_pred = ml_pred.lower().strip()

        if pipe is not None and isinstance(ml_conf, float) and ml_conf >= 0.80:
            final = ml_pred
        else:
            final = 'positive' if (rating >= 4 and vader_label != 'negative') else vader_label

        st.markdown(
            f"""
            <div class="card" style="margin-top:10px;">
                <h2>✨ Result</h2>
                <p><b>VADER:</b> {vader_label} (score {vader:.2f})</p>
                <p><b>ML prediction:</b> {ml_pred} (confidence {ml_conf:.2f})</p>
                <h3 style="color:#d81b60;">Final label: {final.upper()}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.markdown("Tips: Upload small sample files to test quickly. For full dataset analysis, precompute ml_pred and hybrid_label with `python -m src.hybrid` to speed the dashboard.")
