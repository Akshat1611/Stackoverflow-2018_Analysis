import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="Developer Insights Dashboard", layout="wide")
st.title("🚀 Developer Insights Dashboard")
st.write("A unified dashboard for developer clustering, trends, and segmentation.")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    st.success("File uploaded successfully!")
    st.write("### 📄 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # TABS
    # -------------------------------
    tab1, tab2, tab3 = st.tabs([
        "📌 Language-Based Clustering",
        "📈 Language Growth Analysis",
        "👥 DevType Segmentation"
    ])

    # =====================================================
    # TAB 1 — LANGUAGE BASED CLUSTERING
    # =====================================================
    with tab1:
        st.header("🔢 Developer Clustering using Language + Experience")

        lang_col = st.selectbox("Column: Languages Worked With", df.columns,
                                index=df.columns.get_loc("LanguageWorkedWith") if "LanguageWorkedWith" in df.columns else 0)

        years_col = st.selectbox("Column: Years Coding", df.columns,
                                 index=df.columns.get_loc("YearsCoding") if "YearsCoding" in df.columns else 0)

        age_col = st.selectbox("Column: Age", df.columns,
                               index=df.columns.get_loc("Age_clean") if "Age_clean" in df.columns else 0)

        # Process language list
        cluster_df = df[[lang_col, years_col, age_col]].copy()
        cluster_df[lang_col] = cluster_df[lang_col].astype(str).str.split(";")

        mlb = MultiLabelBinarizer()
        lang_df = pd.DataFrame(mlb.fit_transform(cluster_df[lang_col]), columns=mlb.classes_)
        numeric_df = cluster_df[[years_col, age_col]].fillna(0)

        X = pd.concat([lang_df, numeric_df], axis=1)
        scaler = StandardScaler()
        X[[years_col, age_col]] = scaler.fit_transform(numeric_df)

        k = st.slider("Choose number of clusters", 2, 10, 4)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X)

        st.write("### Clustered Data Sample")
        st.dataframe(df[[lang_col, years_col, age_col, "Cluster"]].head())

        # PCA Visualization
        st.subheader("📉 PCA 2D Visualization")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)
        df["PCA1"], df["PCA2"] = pca_result[:, 0], pca_result[:, 1]

        fig, ax = plt.subplots(figsize=(10, 5))
        scatter = ax.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"], alpha=0.7)
        ax.set_title("Clusters in PCA Space")
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")

        st.pyplot(fig)

    # =====================================================
    # TAB 2 — GROWTH RATE ANALYSIS
    # =====================================================
    with tab2:
        st.header("📈 Programming Language Growth (Current vs Future)")

        curr_lang = st.selectbox("Column: Current Languages", df.columns,
                                 index=df.columns.get_loc("LanguageWorkedWith") if "LanguageWorkedWith" in df.columns else 0)

        future_lang = st.selectbox("Column: Future Desired Languages", df.columns,
                                   index=df.columns.get_loc("LanguageDesireNextYear") if "LanguageDesireNextYear" in df.columns else 0)

        df[curr_lang] = df[curr_lang].astype(str).str.split(";")
        df[future_lang] = df[future_lang].astype(str).str.split(";")

        current = df.explode(curr_lang)[curr_lang].value_counts()
        future = df.explode(future_lang)[future_lang].value_counts()

        skills = pd.concat([current, future], axis=1)
        skills.columns = ["CurrentUse", "FutureInterest"]
        skills.fillna(0, inplace=True)

        skills["GrowthRate"] = (
            skills["FutureInterest"] - skills["CurrentUse"]
        ) / skills["CurrentUse"].replace(0, 1)

        skills_sorted = skills.sort_values("GrowthRate", ascending=False)
        skills_sorted = skills_sorted.drop(index="nan", errors="ignore")

        top_n = st.slider("Select top languages", 10, 50, 20)
        top_skills = skills_sorted.head(top_n)

        st.write(f"### Top {top_n} Languages by Growth Rate")
        st.dataframe(top_skills)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(top_skills["CurrentUse"], top_skills["GrowthRate"],
                   s=top_skills["FutureInterest"] * 2, alpha=0.7)

        for lang in top_skills.index:
            ax.text(
                top_skills.loc[lang, "CurrentUse"] + 30,
                top_skills.loc[lang, "GrowthRate"],
                lang,
                fontsize=9
            )

        ax.set_xlabel("Current Use")
        ax.set_ylabel("Growth Rate")
        ax.set_title("Language Growth Rate vs Current Popularity")
        ax.axhline(0, color="gray", linestyle="--")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    # =====================================================
    # TAB 3 — DEVTYPE SEGMENTATION
    # =====================================================
    with tab3:
        st.header("👥 Developer Segmentation based on DevType Responses")

        devtype_col = st.selectbox("Column: DevType", df.columns,
                                   index=df.columns.get_loc("DevType") if "DevType" in df.columns else 0)

        df[devtype_col] = df[devtype_col].astype(str).str.split(";")

        mlb = MultiLabelBinarizer()
        devtype_encoded = mlb.fit_transform(df[devtype_col])
        devtype_df = pd.DataFrame(devtype_encoded, columns=mlb.classes_)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(devtype_df)

        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df["DevCluster"] = kmeans.fit_predict(X_scaled)

        cluster_mapping = {
            0: "Core Software Engineering",
            1: "Students / Early Career",
            2: "Data Science & ML",
            3: "DevOps / Cloud / Systems",
            4: "Mobile / UI/UX / Creative Tech"
        }

        df["ClusterLabel"] = df["DevCluster"].map(cluster_mapping)

        st.write("### Developer Count in Each Segment")
        cluster_totals = df["ClusterLabel"].value_counts()
        st.dataframe(cluster_totals)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(cluster_totals.index, cluster_totals.values, color="skyblue")

        ax.set_title("Developer Segmentation by DevType")
        ax.set_xlabel("Number of Developers")
        ax.set_ylabel("Cluster Label")

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 50, bar.get_y() + bar.get_height()/2, str(width), va="center")

        st.pyplot(fig)

else:
    st.info("📤 Please upload a CSV file to begin.")

