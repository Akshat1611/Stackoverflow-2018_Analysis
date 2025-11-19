import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO

# plotting
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Hackathon Data Studio", layout="wide", initial_sidebar_state="expanded")

# ---- Styling ----
st.markdown("""
<style>
#MainMenu {visibility: hidden;} 
footer {visibility: hidden;} 
header {visibility: hidden;} 
.block-container{padding:1.2rem 1.5rem}
</style>
""", unsafe_allow_html=True)

# ---- Helper functions ----
@st.cache_data
def load_csv(path_or_buffer):
    df = pd.read_csv(path_or_buffer, low_memory=False)
    return df

@st.cache_data
def to_csv_bytes(df):
    out = BytesIO()
    df.to_csv(out, index=False)
    out.seek(0)
    return out

# Try to find a default dataset in common locations (the uploaded hackathon file may have written results)
def find_default_dataset():
    candidates = [
        "cleaned_survey_one.csv",
        "cleaned_survey.csv",
        "/mnt/data/cleaned_survey_one.csv",
        "/mnt/data/cleaned_survey.csv",
        "/mnt/data/cleaned_survey_pakka.csv",
        "/mnt/data/survey_results_public.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# ---- Sidebar ----
st.sidebar.title("Data & Settings")
upload = st.sidebar.file_uploader("Upload CSV file (or use default)", type=["csv"]) 

default_path = find_default_dataset()
use_default = False
if (not upload) and default_path:
    use_default = st.sidebar.checkbox(f"Load default dataset ({os.path.basename(default_path)})", value=True)

# quick display options
st.sidebar.markdown("---")
show_preview = st.sidebar.checkbox("Show data preview", value=True)
show_stats = st.sidebar.checkbox("Show summary statistics", value=True)

# clustering settings
st.sidebar.markdown("---")
st.sidebar.subheader("Clustering")
cluster_cols_input = st.sidebar.text_input("Columns (comma separated) used for clustering", value="YearsCoding,Age_clean")
n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=8, value=4)

# ---- Main ----
st.title("✨ Hackathon — Beautiful Streamlit Interface")
st.caption("Interactive exploration, clustering and quick EDA for your hackathon CSVs")

# Load data
df = None
if upload:
    with st.spinner("Loading uploaded file..."):
        df = load_csv(upload)
elif use_default and default_path:
    with st.spinner(f"Loading {default_path}..."):
        df = load_csv(default_path)
else:
    st.info("No dataset loaded. Upload a CSV on the left or enable the default dataset if available.")

if df is not None:
    st.session_state["_df_rows"] = len(df)

    # top metrics
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing cells", df.isna().sum().sum())
    col4.metric("Unique respondents (approx)", df.index.nunique())

    if show_preview:
        st.subheader("Data preview")
        st.dataframe(df.head(100))

    if show_stats:
        st.subheader("Quick summary statistics")
        num = df.select_dtypes(include=[np.number])
        if not num.empty:
            st.dataframe(num.describe().T)
        else:
            st.info("No numeric columns detected for summary statistics.")

    # Column selector for plots
    st.markdown("---")
    st.subheader("Visualizations")
    cols = df.columns.tolist()
    left, right = st.columns(2)
    with left:
        x_col = st.selectbox("X axis (for scatter / histogram)", options=cols, index=0)
    with right:
        y_col = st.selectbox("Y axis (for scatter)", options=cols, index=min(1, len(cols)-1))

    plot_type = st.radio("Plot type", ["Scatter","Histogram","Box","Bar"], index=0)

    fig = None
    try:
        if plot_type == "Scatter":
            fig = px.scatter(df, x=x_col, y=y_col, hover_data=cols[:6], title=f"Scatter: {x_col} vs {y_col}")
        elif plot_type == "Histogram":
            fig = px.histogram(df, x=x_col, nbins=40, title=f"Histogram: {x_col}")
        elif plot_type == "Box":
            fig = px.box(df, x=x_col, y=y_col, title=f"Box: {x_col} by {y_col}")
        elif plot_type == "Bar":
            vc = df[x_col].value_counts().nlargest(50).reset_index()
            vc.columns = [x_col, 'count']
            fig = px.bar(vc, x=x_col, y='count', title=f"Bar: {x_col} top categories")
    except Exception as e:
        st.error(f"Could not create plot: {e}")

    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap for numeric columns
    if st.checkbox("Show correlation heatmap for numeric cols"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, title="Correlation matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation matrix.")

    # Clustering
    st.markdown("---")
    st.subheader("K-Means clustering (interactive)")
    cluster_cols = [c.strip() for c in cluster_cols_input.split(",") if c.strip() in df.columns]
    if not cluster_cols:
        st.warning("No valid cluster columns found in the dataset. Update the columns in the sidebar.")
    else:
        st.write("Using columns:", cluster_cols)
        data_for_cluster = df[cluster_cols].select_dtypes(include=[np.number]).fillna(0)
        if data_for_cluster.empty:
            st.warning("Selected clustering columns contain no numeric data after selection.")
        else:
            try:
                scaler = StandardScaler()
                Xs = scaler.fit_transform(data_for_cluster)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(Xs)
                df["cluster_label"] = labels

                # show cluster counts
                st.write(df["cluster_label"].value_counts().sort_index())

                # PCA for 2D projection
                if Xs.shape[1] >= 2:
                    pca = PCA(n_components=2, random_state=42)
                    proj = pca.fit_transform(Xs)
                    proj_df = pd.DataFrame(proj, columns=["pca1","pca2"]) 
                    proj_df["cluster"] = labels
                    fig_pca = px.scatter(proj_df, x="pca1", y="pca2", color=proj_df["cluster"].astype(str),
                                         title="PCA projection of clusters", hover_data=["cluster"])
                    st.plotly_chart(fig_pca, use_container_width=True)

            except Exception as e:
                st.error(f"Clustering failed: {e}")

    # Language filter (if LanguageWorkedWith column exists)
    if "LanguageWorkedWith" in df.columns:
        st.markdown("---")
        st.subheader("Language filter & counts")
        # build language list
        lang_series = df["LanguageWorkedWith"].dropna().astype(str)
        languages = set()
        for entry in lang_series:
            for lang in entry.split(";"):
                languages.add(lang.strip())
        languages = sorted([l for l in languages if l])
        sel_lang = st.multiselect("Pick languages to filter rows (OR behavior)", options=languages, max_selections=8)
        if sel_lang:
            mask = lang_series.apply(lambda s: any([(sl in s.split(";")) for sl in sel_lang]))
            filtered = df[mask.values]
            st.write(f"Filtered rows: {filtered.shape[0]}")
            st.dataframe(filtered.head(50))

    # Download cleaned/downloaded dataset
    st.markdown("---")
    st.subheader("Export / Download")
    if st.button("Download current dataframe as CSV"):
        csv_bytes = to_csv_bytes(df)
        st.download_button("Click to download CSV", data=csv_bytes, file_name="dataset_export.csv", mime="text/csv")

    # Try to integrate existing preprocessing script if present
    st.markdown("---")
    st.subheader("Optional: run built-in cleaning from /mnt/data/hackathon.py if available")
    if os.path.exists("/mnt/data/hackathon.py"):
        if st.button("Run /mnt/data/hackathon.py -> clean and reload"):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("hackathon_module", "/mnt/data/hackathon.py")
                hack = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(hack)
                st.success("Ran hackathon.py (if it exposes functions they executed). Please re-load dataset from sidebar if a new file was written.")
            except Exception as e:
                st.error(f"Running hackathon.py failed: {e}")

    # final notes
    st.markdown("---")
    st.info("Tip: Use the sidebar to adjust clustering columns, toggle preview/stats, or upload a different CSV. Enjoy exploring!")

else:
    st.markdown("<div style='text-align:center'><img src='https://images.unsplash.com/photo-1508780709619-79562169bc64?q=80&w=1400&auto=format&fit=crop&crop=entropy&s=7b5d5d2b5a2d3c4d11f9b7e3e3c9ce8a' style='max-width:90%; border-radius:12px' /></div>", unsafe_allow_html=True)
    st.write("Welcome — upload your survey CSV on the left to get started. This app includes interactive plotting, clustering, PCA projection, and an optional integration with /mnt/data/hackathon.py if present.")

# EOF
