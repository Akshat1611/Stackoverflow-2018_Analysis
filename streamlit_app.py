import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------
st.set_page_config(page_title="Developer Clustering App", layout="wide")
st.title("👨‍💻 Developer Clustering App")
st.write("Upload your cleaned StackOverflow-like survey dataset to generate clusters based on skills and experience.")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)

    st.success("File uploaded successfully!")
    st.write("### Preview of Data")
    st.dataframe(df.head())

    # ---------------------------------------------------
    # SELECT COLUMNS (optional)
    # ---------------------------------------------------
    st.subheader("Select Columns for Clustering")
    languages_col = st.selectbox("Column with languages:", df.columns, index=df.columns.get_loc("LanguageWorkedWith") if "LanguageWorkedWith" in df.columns else 0)
    years_col = st.selectbox("Column for Years Coding:", df.columns, index=df.columns.get_loc("YearsCoding") if "YearsCoding" in df.columns else 0)
    age_col = st.selectbox("Column for Age:", df.columns, index=df.columns.get_loc("Age_clean") if "Age_clean" in df.columns else 0)

    # ---------------------------------------------------
    # PROCESS DATA
    # ---------------------------------------------------
    st.header("🔧 Processing Data")

    cluster_df = df[[languages_col, years_col, age_col]].copy()

    # Convert language list
    cluster_df[languages_col] = cluster_df[languages_col].astype(str).str.split(";")

    # MultiLabel Binary
    mlb = MultiLabelBinarizer()
    lang_df = pd.DataFrame(
        mlb.fit_transform(cluster_df[languages_col]),
        columns=mlb.classes_
    )

    # Numeric handling
    numeric_df = cluster_df[[years_col, age_col]].fillna(0)

    # Final dataset
    X = pd.concat([lang_df, numeric_df], axis=1)

    # Scale numeric
    scaler = StandardScaler()
    X[[years_col, age_col]] = scaler.fit_transform(numeric_df)

    # ---------------------------------------------------
    # K-Means
    # ---------------------------------------------------
    st.subheader("🔢 K-Means Clustering")

    k = st.slider("Choose number of clusters", 2, 10, 4)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)

    st.write("### Cluster Output (First 10 rows)")
    st.dataframe(df[[languages_col, years_col, age_col, "Cluster"]].head(10))

    # ---------------------------------------------------
    # PCA VISUALIZATION
    # ---------------------------------------------------
    st.subheader("📊 Cluster Visualization (PCA 2D)")

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)

    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"])
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Clusters in 2D PCA Space")

    st.pyplot(fig)

    # ---------------------------------------------------
    # DOWNLOAD
    # ---------------------------------------------------
    st.subheader("📥 Download Updated Dataset")

    csv_output = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV with Clusters",
        data=csv_output,
        file_name="clustered_output.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file to begin.")
