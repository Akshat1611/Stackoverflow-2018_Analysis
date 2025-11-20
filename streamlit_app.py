import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------------
st.set_page_config(page_title="Developer Insights Dashboard", layout="wide")
st.title("🚀 Developer Insights Dashboard")
st.write("A unified dashboard for clustering, segmentation, trend analysis, and ML insights.")

# ---------------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    st.success("File uploaded successfully!")
    st.write("### 📄 Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------------------------------------------
    # TABS
    # ---------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Language-Based Clustering",
        "📈 Language Growth Analysis",
        "👥 DevType Segmentation",
        "⚖ Work–Life Balance ML Analysis"
    ])

    # ==============================================================
    # TAB 1 — Language-Based Clustering
    # ==============================================================
    with tab1:
        st.header("🔢 Developer Clustering using Language + Experience")

        lang_col = st.selectbox("Column: Languages Worked With", df.columns,
                                index=df.columns.get_loc("LanguageWorkedWith") if "LanguageWorkedWith" in df.columns else 0)

        years_col = st.selectbox("Column: Years Coding", df.columns,
                                 index=df.columns.get_loc("YearsCoding") if "YearsCoding" in df.columns else 0)

        age_col = st.selectbox("Column: Age Column", df.columns,
                               index=df.columns.get_loc("Age_clean") if "Age_clean" in df.columns else 0)

        cluster_df = df[[lang_col, years_col, age_col]].copy()
        cluster_df[lang_col] = cluster_df[lang_col].astype(str).str.split(";")

        mlb = MultiLabelBinarizer()
        lang_df = pd.DataFrame(mlb.fit_transform(cluster_df[lang_col]), columns=mlb.classes_)
        numeric_df = cluster_df[[years_col, age_col]].fillna(0)

        X = pd.concat([lang_df, numeric_df], axis=1)
        scaler = StandardScaler()
        X[[years_col, age_col]] = scaler.fit_transform(numeric_df)

        k = st.slider("Choose number of clusters", 2, 10, 4)
        model = KMeans(n_clusters=k, random_state=42)
        df["Cluster"] = model.fit_predict(X)

        st.write("### Sample Cluster Output")
        st.dataframe(df[[lang_col, years_col, age_col, "Cluster"]].head())

        # PCA Visualization
        st.subheader("📉 PCA Visualization (2D)")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)
        df["PCA1"], df["PCA2"] = pca_result[:, 0], pca_result[:, 1]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"], alpha=0.7)
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.set_title("Clusters in PCA Space")
        st.pyplot(fig)

    # ==============================================================
    # TAB 2 — Language Growth Analysis
    # ==============================================================
    with tab2:
        st.header("📈 Programming Language Growth (Current vs Future)")

        curr_col = st.selectbox("Column: Current Languages", df.columns,
                                index=df.columns.get_loc("LanguageWorkedWith") if "LanguageWorkedWith" in df.columns else 0)
        future_col = st.selectbox("Column: Future Desired Languages", df.columns,
                                  index=df.columns.get_loc("LanguageDesireNextYear") if "LanguageDesireNextYear" in df.columns else 0)

        df[curr_col] = df[curr_col].astype(str).str.split(";")
        df[future_col] = df[future_col].astype(str).str.split(";")

        current = df.explode(curr_col)[curr_col].value_counts()
        future = df.explode(future_col)[future_col].value_counts()

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

        st.write(f"### Top {top_n} Languages By Growth Rate")
        st.dataframe(top_skills)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(top_skills["CurrentUse"], top_skills["GrowthRate"],
                   s=top_skills["FutureInterest"] * 2, alpha=0.7)

        for lang in top_skills.index:
            ax.text(
                top_skills.loc[lang, "CurrentUse"] + 30,
                top_skills.loc[lang, "GrowthRate"],
                lang, fontsize=9
            )

        ax.set_xlabel("Current Use Count")
        ax.set_ylabel("Growth Rate")
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_title("Language Growth Rate vs Current Popularity")
        st.pyplot(fig)

    # ==============================================================
    # TAB 3 — DevType Segmentation
    # ==============================================================
    with tab3:
        st.header("👥 Developer Segmentation using DevType")

        dev_col = st.selectbox("Column: DevType", df.columns,
                               index=df.columns.get_loc("DevType") if "DevType" in df.columns else 0)

        df[dev_col] = df[dev_col].astype(str).str.split(";")

        mlb = MultiLabelBinarizer()
        dev_encoded = mlb.fit_transform(df[dev_col])
        dev_df = pd.DataFrame(dev_encoded, columns=mlb.classes_)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(dev_df)

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

        totals = df["ClusterLabel"].value_counts()
        st.dataframe(totals)

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(totals.index, totals.values, color="skyblue")

        for bar in bars:
            ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                    str(bar.get_width()), va='center')

        ax.set_title("Developer Segmentation by DevType")
        ax.set_xlabel("Number of Developers")
        st.pyplot(fig)

    # ==============================================================
    # TAB 4 — Work–Life Balance ML Feature Importance
    # ==============================================================
    with tab4:
        st.header("⚖ Work–Life Balance Feature Importance (Random Forest)")

        st.write("Parses hour ranges, converts satisfaction to numeric, and shows ML feature importance.")

        # Range parsing function
        def parse_range(value):
            if pd.isna(value):
                return np.nan

            value = str(value).lower()

            match = re.findall(r"(\d+)\s*-\s*(\d+)", value)
            if match:
                a, b = map(int, match[0])
                return (a + b) / 2

            single = re.findall(r"(\d+)", value)
            if single:
                return float(single[0])

            if "less" in value:
                return 0.5

            more = re.findall(r"more than (\d+)", value)
            if more:
                return float(more[0]) + 2

            return np.nan

        cols = ["WorkWeekHrs", "HoursComputer", "HoursOutside",
                "SkipMeals", "Exercise", "JobSatisfaction"]

        use_cols = [c for c in cols if c in df.columns]
        st.write("Using columns:", use_cols)

        wlb = df[use_cols].copy()

        for col in wlb.columns:
            if col != "JobSatisfaction":
                wlb[col] = wlb[col].apply(parse_range)

        le = LabelEncoder()
        wlb["JobSatisfaction"] = le.fit_transform(wlb["JobSatisfaction"].astype(str))

        wlb = wlb.replace([np.inf, -np.inf], np.nan)
        wlb = wlb.dropna()

        st.write("Rows after cleaning:", len(wlb))

        st.write("### Correlation with Job Satisfaction")
        st.dataframe(wlb.corr()["JobSatisfaction"].sort_values(ascending=False))

        X = wlb.drop("JobSatisfaction", axis=1)
        y = wlb["JobSatisfaction"]

        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.write("### Feature Importance")
        st.dataframe(importance)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(importance["Feature"], importance["Importance"], color="skyblue")
        ax.set_title("Work–Life Balance Feature Importance")
        ax.set_xlabel("Importance")
        ax.invert_yaxis()
        st.pyplot(fig)

else:
    st.info("📤 Please upload a CSV dataset to begin.")
