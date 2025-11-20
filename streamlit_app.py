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
st.write("A powerful dashboard for clustering, sentiment analysis, ML insights, and developer segmentation.")

# ---------------------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your cleaned CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    st.success("File uploaded successfully!")
    st.write("### 📄 Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------------------------------------------
    # TABS (All Features)
    # ---------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📌 Language-Based Clustering",
        "📈 Language Growth Analysis",
        "👥 DevType Segmentation",
        "⚖ Work–Life Balance ML Analysis",
        "🧠 AI Sentiment Categorization"
    ])

    # ==============================================================
    # TAB 1 — LANGUAGE-BASED CLUSTERING
    # ==============================================================
    with tab1:
        st.header("🔢 Developer Clustering using Language + Experience")

        lang_col = st.selectbox("Column: Languages Worked With", df.columns)
        years_col = st.selectbox("Column: Years Coding", df.columns)
        age_col = st.selectbox("Column: Age Column", df.columns)

        cluster_df = df[[lang_col, years_col, age_col]].copy()
        cluster_df[lang_col] = cluster_df[lang_col].astype(str).str.split(";")

        mlb = MultiLabelBinarizer()
        lang_df = pd.DataFrame(mlb.fit_transform(cluster_df[lang_col]), columns=mlb.classes_)

        numeric_df = cluster_df[[years_col, age_col]].fillna(0)

        X = pd.concat([lang_df, numeric_df], axis=1)
        scaler = StandardScaler()
        X[[years_col, age_col]] = scaler.fit_transform(numeric_df)

        k = st.slider("Choose number of clusters", 2, 10, 4)
        km = KMeans(n_clusters=k, random_state=42)
        df["Cluster"] = km.fit_predict(X)

        st.write("### Sample Cluster Output")
        st.dataframe(df[[lang_col, years_col, age_col, "Cluster"]].head())

        # PCA Plot
        st.subheader("📉 PCA Visualization (2D)")
        pca = PCA(n_components=2)
        comp = pca.fit_transform(X)
        df["PCA1"], df["PCA2"] = comp[:, 0], comp[:, 1]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"], alpha=0.7)
        ax.set_title("Clusters in PCA Space")
        st.pyplot(fig)

    # ==============================================================
    # TAB 2 — LANGUAGE GROWTH ANALYSIS
    # ==============================================================
    with tab2:
        st.header("📈 Programming Language Growth (Current vs Future)")

        curr = st.selectbox("Current Languages Column", df.columns)
        future = st.selectbox("Future Desired Languages Column", df.columns)

        df[curr] = df[curr].astype(str).str.split(";")
        df[future] = df[future].astype(str).str.split(";")

        current = df.explode(curr)[curr].value_counts()
        future_use = df.explode(future)[future].value_counts()

        skills = pd.concat([current, future_use], axis=1)
        skills.columns = ["CurrentUse", "FutureInterest"]
        skills.fillna(0, inplace=True)

        skills["GrowthRate"] = (skills["FutureInterest"] - skills["CurrentUse"]) / skills["CurrentUse"].replace(0, 1)

        skills.sort_values("GrowthRate", ascending=False, inplace=True)
        skills = skills.drop(index="nan", errors="ignore")

        top_n = st.slider("Top languages to display", 10, 50, 20)
        top_langs = skills.head(top_n)

        st.dataframe(top_langs)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(top_langs["CurrentUse"], top_langs["GrowthRate"],
                   s=top_langs["FutureInterest"] * 2)
        for lang in top_langs.index:
            ax.text(top_langs.loc[lang, "CurrentUse"] + 30,
                    top_langs.loc[lang, "GrowthRate"],
                    lang, fontsize=9)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_title("Growth Rate vs Current Popularity")
        st.pyplot(fig)

    # ==============================================================
    # TAB 3 — DEVTYPE SEGMENTATION
    # ==============================================================
    with tab3:
        st.header("👥 Developer Segmentation using DevType")

        dev = st.selectbox("DevType Column", df.columns)

        df[dev] = df[dev].astype(str).str.split(";")

        mlb = MultiLabelBinarizer()
        encoded = mlb.fit_transform(df[dev])
        dev_df = pd.DataFrame(encoded, columns=mlb.classes_)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(dev_df)

        km = KMeans(n_clusters=5, random_state=42, n_init=10)
        df["DevCluster"] = km.fit_predict(scaled)

        mapping = {
            0: "Core Software Engineering",
            1: "Students / Early Career",
            2: "Data Science & ML",
            3: "DevOps / Cloud / Systems",
            4: "Mobile / UI/UX / Creative Tech"
        }

        df["ClusterLabel"] = df["DevCluster"].map(mapping)
        totals = df["ClusterLabel"].value_counts()

        st.dataframe(totals)

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(totals.index, totals.values, color="skyblue")

        for bar in bars:
            ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                    str(bar.get_width()), va='center')
        ax.set_title("Developer Segmentation by DevType")
        st.pyplot(fig)

    # ==============================================================
    # TAB 4 — WORK–LIFE BALANCE ML ANALYSIS
    # ==============================================================
    with tab4:
        st.header("⚖ Work–Life Balance Feature Importance (Random Forest)")

        cols = ["WorkWeekHrs", "HoursComputer", "HoursOutside",
                "SkipMeals", "Exercise", "JobSatisfaction"]

        use = [c for c in cols if c in df.columns]
        wlb = df[use].copy()

        def parse_range(value):
            if pd.isna(value):
                return np.nan
            value = str(value).lower()
            match = re.findall(r"(\d+)\s*-\s*(\d+)", value)
            if match:
                a, b = map(int, match[0])
                return (a + b) / 2
            num = re.findall(r"(\d+)", value)
            if num:
                return float(num[0])
            if "less" in value:
                return 0.5
            if "more than" in value:
                return float(re.findall(r"(\d+)", value)[0]) + 2
            return np.nan

        for col in wlb.columns:
            if col != "JobSatisfaction":
                wlb[col] = wlb[col].apply(parse_range)

        le = LabelEncoder()
        wlb["JobSatisfaction"] = le.fit_transform(wlb["JobSatisfaction"].astype(str))

        wlb = wlb.replace([np.inf, -np.inf], np.nan).dropna()

        st.write("### Correlations with JobSatisfaction")
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
        ax.invert_yaxis()
        ax.set_title("Work–Life Balance Feature Importance")
        st.pyplot(fig)

    # ==============================================================
    # TAB 5 — AI SENTIMENT CATEGORIZATION
    # ==============================================================
    with tab5:
        st.header("🧠 AI Sentiment / Opinion Categorization")

        ai_cols = ["AIDangerous", "AIInteresting", "AIResponsible", "AIFuture"]
        ai_cols = [c for c in ai_cols if c in df.columns]

        if len(ai_cols) == 0:
            st.error("Dataset missing required AI opinion columns.")
        else:
            ai = df[ai_cols].copy()

            for col in ai_cols:
                ai[col] = ai[col].astype(str).lower().str.strip()

            def danger_level(text):
                if "superintelligence" in text or "singularity" in text:
                    return "High Risk Concern"
                if "making important decisions" in text:
                    return "Decision-Making Concern"
                if "automation" in text:
                    return "Job Automation Concern"
                return "Unknown / Missing"

            def interest_type(text):
                if "superintelligence" in text:
                    return "Superintelligence Curiosity"
                if "making important decisions" in text:
                    return "Decision AI Interest"
                if "versus human" in text:
                    return "Ethical Comparison"
                return "Unknown / Missing"

            def responsible_type(text):
                if "people creating" in text:
                    return "Developers Responsible"
                if "regulatory" in text or "national" in text:
                    return "Government Responsible"
                return "Unknown / Missing"

            def future_sent(text):
                if "worried" in text:
                    return "Negative (Danger Concern)"
                if "excited" in text:
                    return "Positive (Optimistic)"
                if "haven't thought" in text:
                    return "Neutral"
                return "Unknown / Missing"

            ai["DangerCategory"] = ai["AIDangerous"].apply(danger_level)
            ai["InterestCategory"] = ai["AIInteresting"].apply(interest_type)
            ai["ResponsibleCategory"] = ai["AIResponsible"].apply(responsible_type)
            ai["FutureSentiment"] = ai["AIFuture"].apply(future_sent)

            st.subheader("📊 AI Danger Categories")
            st.dataframe(ai["DangerCategory"].value_counts())

            st.subheader("📊 AI Interest Categories")
            st.dataframe(ai["InterestCategory"].value_counts())

            st.subheader("📊 Who Is Responsible for AI?")
            st.dataframe(ai["ResponsibleCategory"].value_counts())

            st.subheader("📊 AI Future Sentiment")
            st.dataframe(ai["FutureSentiment"].value_counts())

            # Visualization
            fig, ax = plt.subplots(2, 2, figsize=(14, 10))
            ai["DangerCategory"].value_counts().plot(kind="bar", ax=ax[0][0], color="tomato")
            ai["InterestCategory"].value_counts().plot(kind="bar", ax=ax[0][1], color="gold")
            ai["ResponsibleCategory"].value_counts().plot(kind="bar", ax=ax[1][0], color="skyblue")
            ai["FutureSentiment"].value_counts().plot(kind="bar", ax=ax[1][1], color="green")

            plt.tight_layout()
            st.pyplot(fig)

else:
    st.info("📤 Please upload a CSV dataset to begin.")
