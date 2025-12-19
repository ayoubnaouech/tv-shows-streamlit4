import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="TV Shows Popularity Analysis",
    layout="wide"
)

st.title("📺 TV Shows Popularity & User Behavior Analysis")

st.markdown("""
Interactive analysis of TV show popularity using **genres, user age,
countries, and temporal trends**, with synthetic but realistic
demographic simulation.
""")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("10k_Poplar_Tv_Shows_with_users_updated6.csv")

    # Parse list-like columns
    df["genre_names"] = df["genre_names"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df["origin_country"] = df["origin_country"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Date handling
    df["first_air_date"] = pd.to_datetime(df["first_air_date"], errors="coerce")
    df["year"] = df["first_air_date"].dt.year
    df["decade"] = (df["year"] // 10) * 10

    # Age groups (if not already present)
    if "age_group" not in df.columns:
        df["age_group"] = pd.cut(
            df["user_age"],
            bins=[12, 17, 24, 34, 44, 54, 70],
            labels=[
                "Teen (13–17)", "Young Adult (18–24)", "Adult (25–34)",
                "Mid Adult (35–44)", "Older Adult (45–54)", "Senior (55–70)"
            ]
        )

    return df


df = load_data()
st.success(f"Dataset loaded successfully — {df.shape[0]} TV shows")

# ==============================
# SIDEBAR FILTERS
# ==============================
st.sidebar.header("🔎 Global Filters")

# Age group filter
selected_age_groups = st.sidebar.multiselect(
    "Select Age Groups",
    options=df["age_group"].dropna().unique(),
    default=df["age_group"].dropna().unique()
)

# Genre filter
all_genres = sorted(
    {g for sub in df["genre_names"] if isinstance(sub, list) for g in sub}
)

selected_genres = st.sidebar.multiselect(
    "Select Genres",
    options=all_genres,
    default=all_genres
)

# Apply filters (IMPORTANT: do not overwrite df)
df_filt = df[df["age_group"].isin(selected_age_groups)]

df_filt = df_filt[
    df_filt["genre_names"].apply(
        lambda genres: any(g in genres for g in selected_genres)
        if isinstance(genres, list) else False
    )
]

# ==============================
# OVERVIEW METRICS
# ==============================
st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Shows", df_filt.shape[0])
col2.metric("Average Vote", round(df_filt["vote_average"].mean(), 2))
col3.metric("Average Popularity", round(df_filt["popularity"].mean(), 2))

# ==============================
# USER AGE DISTRIBUTION
# ==============================
st.header("👤 User Age Distribution")

fig, ax = plt.subplots()
ax.hist(df_filt["user_age"], bins=20)
ax.set_xlabel("User Age")
ax.set_ylabel("Count")
st.pyplot(fig)

# ==============================
# POPULARITY DISTRIBUTION
# ==============================
st.header("⭐ Popularity Distribution")

fig, ax = plt.subplots()
ax.hist(df_filt["popularity"], bins=30)
ax.set_xlabel("Popularity")
ax.set_ylabel("Number of Shows")
st.pyplot(fig)

# ==============================
# VOTE RELATIONSHIP
# ==============================
st.header("🗳 Vote Average vs Vote Count")

fig, ax = plt.subplots()
ax.scatter(
    df_filt["vote_average"],
    df_filt["vote_count"],
    alpha=0.5
)
ax.set_xlabel("Vote Average")
ax.set_ylabel("Vote Count")
st.pyplot(fig)

# ==============================
# CORRELATION HEATMAP
# ==============================
st.header("📈 Correlation Heatmap")

numeric_cols = ["popularity", "vote_average", "vote_count", "user_age"]
corr = df_filt[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# ==============================
# TOP GENRES
# ==============================
st.header("🎭 Top Genres")

genre_counts = (
    df_filt.explode("genre_names")["genre_names"]
    .value_counts()
    .head(10)
)

fig, ax = plt.subplots()
genre_counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Genre")
ax.set_ylabel("Number of Shows")
st.pyplot(fig)

# ==============================
# SHOWS OVER TIME
# ==============================
st.header("📅 Number of Shows per Year")

shows_per_year = (
    df_filt["year"]
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots()
ax.plot(shows_per_year.index, shows_per_year.values)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Shows")
st.pyplot(fig)

# ==============================
# GENRE POPULARITY BY AGE GROUP
# ==============================
st.header("🔥 Genre Popularity by Age Group")

df_exp = df_filt.explode("genre_names")

pivot = pd.pivot_table(
    df_exp,
    values="popularity",
    index="genre_names",
    columns="age_group",
    aggfunc="mean"
)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, cmap="YlGnBu", ax=ax)
st.pyplot(fig)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("""
**Project:** TV Shows Popularity Analysis  
**Tech Stack:** Python · Pandas · Streamlit · Matplotlib · Seaborn  
**Methodology:** Content-aware synthetic demographics
""")
