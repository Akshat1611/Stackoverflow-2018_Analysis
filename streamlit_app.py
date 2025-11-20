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
# SAFE TEXT CLEANER (Fixes all .str errors)
# ---------------------------------------------------------------
def safe_text(x):
    if isinstance(x, list):
        return " ".join(str(i) for i in x)
    if pd.isna(x):
        return ""
    return str(x)

# ---------------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------------
st.set_page_config(page_title="Developer Insights Dashboard", layout="wide")
st.title("🚀 Developer Insights Dashboard")

# ---------------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------------
uploaded_file = st.file_uploader("Upload cleaned CSV", type=["csv"])

if not uploaded_file:
    st.info("📤 Please upload a CSV dataset to begin.")
    st.stop()

df = pd.read_csv(uploaded_file, low_memory=False)
st.success("File uploaded successfully!")
st.dataframe(df.head())

# ---------------------------------------------------------------
# TABS
# ---------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Language Clustering",
    "📈 Language Growth",
    "👥 DevType Segmentation",
    "⚖ Work–Life ML",
    "🧠 AI Sentiment"
])


# ==============================================================
# TAB 1 — LANGUAGE CLUSTERING
# ==============================================================
with tab1:
    st.header("🔢 Developer Clustering")

    lang_col = st.selectbox("Languages Column", df.columns)
    years_col = st.selectbox("Years Coding Column", df.columns)
    age_col = st.selectbox("Age Column", df.columns)

    temp = df[[lang_col, years_col, age_col]].copy()
    temp[lang_col] = temp[lang_col].apply(lambda x: safe_text(x).split(";"))

    mlb = MultiLabelBinarizer()
    lang_df = pd.DataFrame(mlb.fit_transform(temp[lang_col]), columns=mlb.classes_)

    num_df = temp[[years_col, age_col]].fillna(0)

    X = pd.concat([lang_df, num_df], axis=1)
    scaler = StandardScaler()
    X[[years_col, age_col]] = scaler.fit_transform(num_df)

    k = st.slider("Clusters", 2, 10, 4)
    km = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = km.fit_predict(X)

    st.dataframe(df[[lang_col, years_col, age_col, "Cluster"]].head())

    pca = PCA(n_components=2)
    comp = pca.fit_transform(X)
    df["PCA1"], df["PCA2"] = comp[:, 0], comp[:, 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"])
    st.pyplot(fig)


# ==============================================================
# TAB 2 — LANGUAGE GROWTH
# ==============================================================
with tab2:
    st.header("📈 Language Growth Forecast")

    curr = st.selectbox("Current Languages", df.columns)
    future = st.selectbox("Future Desired Languages", df.columns)

    df[curr] = df[curr].apply(lambda x: safe_text(x).split(";"))
    df[future] = df[future].apply(lambda x: safe_text(x).split(";"))

    current = df.explode(curr)[curr].value_counts()
    future_use = df.explode(future)[future].value_counts()

    skills = pd.concat([current, future_use], axis=1)
    skills.columns = ["CurrentUse", "FutureInterest"]
    skills.fillna(0, inplace=True)

    skills["GrowthRate"] = (skills["FutureInterest"] - skills["CurrentUse"]) / skills["CurrentUse"].replace(0, 1)
    skills = skills.drop(index="nan", errors="ignore").sort_values("GrowthRate", ascending=False)

    top_n = st.slider("Top Languages", 10, 50, 20)
    top = skills.head(top_n)
    st.dataframe(top)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(top["CurrentUse"], top["GrowthRate"], s=top["FutureInterest"] * 2)
    for lang in top.index:
        ax.text(top.loc[lang, "CurrentUse"] + 5, top.loc[lang, "GrowthRate"], lang, fontsize=8)
    st.pyplot(fig)


# ==============================================================
# TAB 3 — DEVTYPE SEGMENTATION
# ==============================================================
with tab3:
    st.header("👥 Developer Type Segmentation")

    dev = st.selectbox("DevType Column", df.columns)

    df[dev] = df[dev].apply(lambda x: safe_text(x).split(";"))

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
        3: "DevOps / Cloud",
        4: "Mobile / UI/UX"
    }
    df["ClusterLabel"] = df["DevCluster"].map(mapping)

    totals = df["ClusterLabel"].value_counts()
    st.dataframe(totals)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(totals.index, totals.values)
    st.pyplot(fig)


# ==============================================================
# TAB 4 — WORK–LIFE BALANCE ML
# ==============================================================
with tab4:
    st.header("⚖ Work–Life Balance Feature Importance")

    cols = ["WorkWeekHrs", "HoursComputer", "HoursOutside",
            "SkipMeals", "Exercise", "JobSatisfaction"]

    use = [c for c in cols if c in df.columns]
    wlb = df[use].copy()

    def parse_range(value):
        text = safe_text(value)
        match = re.findall(r"(\d+)\s*-\s*(\d+)", text)
        if match:
            a, b = map(int, match[0])
            return (a + b) / 2
        nums = re.findall(r"(\d+)", text)
        if nums:
            return float(nums[0])
        if "less" in text:
            return 0.5
        if "more than" in text:
            return float(re.findall(r"(\d+)", text)[0]) + 2
        return np.nan

    for col in wlb.columns:
        if col != "JobSatisfaction":
            wlb[col] = wlb[col].apply(parse_range)

    le = LabelEncoder()
    wlb["JobSatisfaction"] = le.fit_transform(wlb["JobSatisfaction"].astype(str))

    wlb = wlb.replace([np.inf, -np.inf], np.nan).dropna()

    X = wlb.drop("JobSatisfaction", axis=1)
    y = wlb["JobSatisfaction"]

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.dataframe(importance)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance["Feature"], importance["Importance"])
    ax.invert_yaxis()
    st.pyplot(fig)


# ==============================================================
# TAB 5 — AI SENTIMENT
# ==============================================================
with tab5:
    st.header("🧠 AI Sentiment Categorization")

    ai_cols = ["AIDangerous", "AIInteresting", "AIResponsible", "AIFuture"]
    ai_cols = [c for c in ai_cols if c in df.columns]

    if len(ai_cols) == 0:
        st.error("AI columns not found.")
    else:
        # Convert every value to clean string safely
        ai = df[ai_cols].applymap(lambda x: safe_text(x).lower().strip())

        # ---------- FIX: ENSURE X IS STRING ----------
        def danger(x):
            x = str(x)
            if "superintelligence" in x or "singularity" in x:
                return "High Risk Concern"
            if "making important decisions" in x:
                return "Decision-Making Concern"
            if "automation" in x:
                return "Job Automation Concern"
            return "Unknown / Missing"

        def interest(x):
            x = str(x)
            if "superintelligence" in x:
                return "Superintelligence Curiosity"
            if "making important decisions" in x:
                return "Decision AI Interest"
            if "versus human" in x:
                return "Ethical Comparison"
            return "Unknown / Missing"

        def responsible(x):
            x = str(x)
            if "people creating" in x:
                return "Developers Responsible"
            if "regulatory" in x or "national" in x:
                return "Government Responsible"
            return "Unknown / Missing"

        def future(x):
            x = str(x)
            if "worried" in x:
                return "Negative"
            if "excited" in x:
                return "Positive"
            if "haven't thought" in x:
                return "Neutral"
            return "Unknown / Missing"

        # Apply
        ai["Danger"] = ai["AIDangerous"].apply(danger)
        ai["Interest"] = ai["AIInteresting"].apply(interest)
        ai["Responsible"] = ai["AIResponsible"].apply(responsible)
        ai["Future"] = ai["AIFuture"].apply(future)

        st.dataframe(ai)

        # ---------- Plots ----------
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))

        ai["Danger"].value_counts().plot(kind="bar", ax=ax[0][0], color="red")
        ai["Interest"].value_counts().plot(kind="bar", ax=ax[0][1], color="orange")
        ai["Responsible"].value_counts().plot(kind="bar", ax=ax[1][0], color="skyblue")
        ai["Future"].value_counts().plot(kind="bar", ax=ax[1][1], color="green")

        st.pyplot(fig)
