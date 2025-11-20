import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -------------------------------
# STREAMLIT PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="Developer Insights Dashboard", layout="wide")

st.title("🚀 Developer Insights Dashboard")
st.write("Analyze developer trends using **Clustering** and **Growth Rate Visualization**.")

# -------------------------------
# FILE UPLOADER
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    st.success("File uploaded successfully!")
    st.write("### 📄 Data Preview")
    st.dataframe(df.head())

    # -------------------------------
    # CREATE TABS
    # -------------------------------
    tab1, tab2 = st.tabs(["📌 K-Means Clustering", "📈 Language Growth Visualization"])

    # ======================================================
    # TAB 1 : CLUSTERING
    # ======================================================
    with tab1:
        st.header("🔢 Developer Clustering (K-Means)")

        # Column selection
        st.subheader("Select Columns")

        lang_col = st.selectbox("Column: Languages Worked With", df.columns,
                                index=df.columns.get_loc("LanguageWorkedWith") if "LanguageWorkedWith" in df.columns else 0)

        years_col = st.selectbox("Column: Years Coding", df.columns,
                                 index=df.columns.get_loc("YearsCoding") if "YearsCoding" in df.columns else 0)

        age_col = st.selectbox("Column: Age", df.columns,
                               index=df.columns.get_loc("Age_clean") if "Age_clean" in df.columns else 0)

        # Process language list
        cluster_df = df[[lang_col, years_col, age_col]].copy()
        cluster_df[lang_col] = cluster_df[lang_col].astype(str).str.split(";")

        # Multi-label binarizer
        mlb = MultiLabelBinarizer()
        lang_df = pd.DataFrame(mlb.fit_transform(cluster_df[lang_col]),
                               columns=mlb.classes_)

        numeric_df = cluster_df[[years_col, age_col]].fillna(0)

        X = pd.concat([lang_df, numeric_df], axis=1)

        scaler = StandardScaler()
        X[[years_col, age_col]] = scaler.fit_transform(numeric_df)

        # Select clusters
        k = st.slider("Select number of clusters", 2, 10, 4)

        kmeans = KMeans(n_clusters=k, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X)

        st.write("### 📊 Sample Cluster Output")
        st.dataframe(df[[lang_col, years_col, age_col, "Cluster"]].head(10))

        # PCA Visualization
        st.subheader("📉 PCA Visualization (2D)")

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)
        df["PCA1"], df["PCA2"] = pca_result[:, 0], pca_result[:, 1]

        fig, ax = plt.subplots(figsize=(10, 5))
        scatter = ax.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"], alpha=0.7)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title("Clusters in PCA Space")
        st.pyplot(fig)

        # Download
        csv_output = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Clustered Dataset", csv_output,
                           file_name="clustered_output.csv", mime="text/csv")

    # ======================================================
    # TAB 2 : LANGUAGE GROWTH VISUALIZATION
    # ======================================================
    with tab2:
        st.header("📈 Language Growth: Current vs Future Trends")

        lang_current_col = st.selectbox("Column: Current Languages", df.columns,
                                        index=df.columns.get_loc("LanguageWorkedWith") if "LanguageWorkedWith" in df.columns else 0)

        lang_future_col = st.selectbox("Column: Future Desired Languages", df.columns,
                                       index=df.columns.get_loc("LanguageDesireNextYear") if "LanguageDesireNextYear" in df.columns else 0)

        # Clean language columns
        df[lang_current_col] = df[lang_current_col].astype(str).str.split(";")
        df[lang_future_col] = df[lang_future_col].astype(str).str.split(";")

        # Count frequencies
        current = df.explode(lang_current_col)[lang_current_col].value_counts()
        future = df.explode(lang_future_col)[lang_future_col].value_counts()

        skills = pd.concat([current, future], axis=1)
        skills.columns = ["CurrentUse", "FutureInterest"]
        skills.fillna(0, inplace=True)

        # Growth Rate
        skills["GrowthRate"] = (skills["FutureInterest"] - skills["CurrentUse"]) / skills["CurrentUse"].replace(0, 1)

        skills_sorted = skills.sort_values("GrowthRate", ascending=False)
        skills_sorted = skills_sorted.drop(index="nan", errors="ignore")

        # Top N selector
        top_n = st.slider("Select top languages to display", 10, 50, 20)
        skills_clean = skills_sorted.head(top_n)

        st.write(f"### 🔝 Top {top_n} Languages by Growth Rate")
        st.dataframe(skills_clean)

        # Plot bubble chart
        st.subheader("📊 Growth Rate vs Current Popularity")

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.scatter(
            skills_clean["CurrentUse"],
            skills_clean["GrowthRate"],
            s=skills_clean["FutureInterest"] * 2,
            alpha=0.7
        )

        # Label points
        for lang in skills_clean.index:
            x, y = skills_clean.loc[lang, "CurrentUse"], skills_clean.loc[lang, "GrowthRate"]
            ax.text(x + 30, y, lang, fontsize=9)

        ax.set_xlabel("Current Use Count")
        ax.set_ylabel("Growth Rate")
        ax.set_title("Programming Language Growth Rate vs Current Popularity")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.grid(alpha=0.2)

        st.pyplot(fig)

else:
    st.info("📤 Please upload a CSV file to begin your analysis.")
