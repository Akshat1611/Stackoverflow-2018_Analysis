import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans

# Streamlit layout
st.set_page_config(page_title="Survey Data Cleaner", layout="wide")

st.title("✨ Survey Data Cleaning & Clustering App")
st.write("Upload a CSV file to clean and cluster it in minutes!")

pd.set_option('future.no_silent_downcasting', True)

# -----------------------------------------
# FILE UPLOAD
# -----------------------------------------
uploaded = st.file_uploader("📂 Upload survey_results_public.csv", type=["csv"])

if uploaded:

    df = pd.read_csv(uploaded, low_memory=False)
    st.success("✔ File uploaded successfully")

    st.write("### 🔍 Raw Data Preview")
    st.dataframe(df.head(), width="stretch")

    # -----------------------------------------
    # DATA CLEANING
    # -----------------------------------------
    st.header("🧹 Data Cleaning")

    df = df.reset_index(drop=True)

    # remove duplicates
    df.drop_duplicates(inplace=True)

    # drop columns >70% missing
    missing_percent = df.isna().mean()
    cols_to_drop = missing_percent[missing_percent > 0.70].index
    df.drop(columns=cols_to_drop, inplace=True)

    # YES/NO standardization
    def convert_yes_no(col):
        mapping = {
            "yes": True, "y": True,
            "no": False, "n": False,
            "Yes": True, "Y": True,
            "No": False, "N": False
        }
        col = col.replace(mapping)
        return col.infer_objects(copy=False)

    for col in df.columns:
        series = df[col].astype(str).str.lower()
        if series.isin(["yes", "no", "y", "n"]).any():
            df[col] = convert_yes_no(df[col])

    # Clean Age
    def clean_age(x):
        if isinstance(x, str) and "-" in x:
            x = x.replace("years old", "").strip()
            a, b = x.split("-")
            return (float(a) + float(b)) / 2
        try:
            return float(x)
        except:
            return np.nan

    if "Age" in df.columns:
        df["Age_clean"] = df["Age"].apply(clean_age)

    # Clean salary numbers
    salary_cols = [c for c in df.columns if "Salary" in c]
    for col in salary_cols:
        df[col] = (
            df[col].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean YearsCoding
    def clean_years_coding(x):
        if pd.isna(x):
            return np.nan
        x = str(x).lower().strip()

        if "-" in x:
            try:
                a, b = x.replace("years","").replace("year","").split("-")
                return (float(a)+float(b))/2
            except:
                return np.nan
        
        if "less than" in x:
            return 0.5
        if "more than" in x:
            return 50
        
        x = x.replace("years","").replace("year","").strip()
        try:
            return float(x)
        except:
            return np.nan

    if "YearsCoding" in df.columns:
        df["YearsCoding"] = df["YearsCoding"].apply(clean_years_coding)

    # Fill missing numeric/categorical
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=["object", "bool"]).columns
    df[cat_cols] = (
        df[cat_cols]
        .fillna(df[cat_cols].mode().iloc[0])
        .infer_objects(copy=False)
    )

    st.success("🎉 Cleaning completed successfully!")

    st.write("### ✔ Cleaned Data Preview")
    st.dataframe(df.head(), width="stretch")

    # -----------------------------------------
    # DOWNLOAD CLEANED FILE
    # -----------------------------------------
    cleaned_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Cleaned CSV", cleaned_csv, "cleaned_data.csv")

    # -----------------------------------------
    # CLUSTERING
    # -----------------------------------------
    st.header("🔮 Developer Clustering Engine (Languages + Age + Experience)")

    if st.button("Run Clustering"):

        # ensure dataframe – avoid mismatched lengths
        df = df.reset_index(drop=True)

        cluster_df = df[["LanguageWorkedWith", "YearsCoding", "Age_clean"]].copy()
        cluster_df = cluster_df.reset_index(drop=True)

        # fix missing and ensure consistent format
        cluster_df["LanguageWorkedWith"] = (
            cluster_df["LanguageWorkedWith"]
            .fillna("")
            .astype(str)
            .str.split(";")
        )

        # Language binarization
        mlb = MultiLabelBinarizer()
        lang_df = pd.DataFrame(
            mlb.fit_transform(cluster_df["LanguageWorkedWith"]),
            columns=mlb.classes_
        )

        numeric = cluster_df[["YearsCoding", "Age_clean"]].fillna(0)

        # Final dataset
        X = pd.concat([lang_df, numeric], axis=1)
        X = X.reset_index(drop=True)

        # Scale numeric columns
        scaler = StandardScaler()
        X[["YearsCoding", "Age_clean"]] = scaler.fit_transform(numeric)

        # KMeans
        kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
        df["Cluster"] = kmeans.fit_predict(X)

        st.success("✨ Clustering Completed Successfully!")

        st.write("### Clustered Data Preview")
        st.dataframe(df[["LanguageWorkedWith", "YearsCoding", "Age_clean", "Cluster"]].head(), width="stretch")

        # download clustered
        clustered_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Clustered CSV", clustered_csv, "clustered_data.csv")
