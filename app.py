# app.py
import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="TV Shows Popularity Analysis",
    layout="wide"
)

st.title("📺 TV Shows Popularity & User Behavior Analysis")
st.markdown(
    """
    This interactive application explores **TV show popularity**
    based on **genres, user age, country, and time** using synthetic
    demographic simulation.
    """
)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("10k_Poplar_Tv_Shows_with_users_updated6.csv")

    # Parse list columns safely
    df["genre_names"] = df["genre_names"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df["origin_country"] = df["origin_country"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    df["first_air_date"] = pd.to_datetime(df["first_air_date"], errors="coerce")
    df["year"] = df["first_air_date"].dt.year
    df["decade"] = (df["year"] // 10) * 10

    return df


df = load_data()

st.success(f"Dataset loaded successfully — {df.shape[0]} TV shows")

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("🔍 Filters")

selected_age_group = st.sidebar.multiselect(
    "Select Age Group",
    df["age_group"].dropna().unique(),
    default=df["age_group"].dropna().unique()
)

selected_genres = st.sidebar.multiselect(
    "Select Genres",
    sorted({g for sub in df["genre_names"] for g in sub}),
    default=None
)

filtered_df = df[df["age_group"].isin(selected_age_group)]

if selected_genres:
    filtered_df = filtered_df[
        filtered_df["genre_names"].apply(
            lambda genres: any(g in genres for g in selected_genres)
        )
    ]

# -----------------------------
# Section 1: Age distribution
# -----------------------------
st.header("👤 User Age Distribution")

fig, ax = plt.subplots()
ax.hist(filtered_df["user_age"], bins=20)
ax.set_xlabel("User Age")
ax.set_ylabel("Count")
ax.set_title("Distribution of User Ages")
st.pyplot(fig)

# -----------------------------
# Section 2: Popularity distribution
# -----------------------------
st.header("⭐ Popularity Distribution")

fig, ax = plt.subplots()
ax.hist(filtered_df["popularity"], bins=30)
ax.set_xlabel("Popularity")
ax.set_ylabel("Number of Shows")
st.pyplot(fig)

# -----------------------------
# Section 3: Vote relationship
# -----------------------------
st.header("🗳 Vote Average vs Vote Count")

fig, ax = plt.subplots()
ax.scatter(
    filtered_df["vote_average"],
    filtered_df["vote_count"],
    alpha=0.5
)
ax.set_xlabel("Vote Average")
ax.set_ylabel("Vote Count")
st.pyplot(fig)

# -----------------------------
# Section 4: Correlation heatmap
# -----------------------------
st.header("📊 Correlation Heatmap")

numeric_cols = ["popularity", "vote_average", "vote_count", "user_age"]
corr = filtered_df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# -----------------------------
# Section 5: Top genres
# -----------------------------
st.header("🎭 Top Genres")

genre_counts = (
    filtered_df.explode("genre_names")["genre_names"]
    .value_counts()
    .head(10)
)

fig, ax = plt.subplots()
genre_counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Genre")
ax.set_ylabel("Number of Shows")
st.pyplot(fig)

# -----------------------------
# Section 6: Shows over time
# -----------------------------
st.header("📈 Number of Shows per Year")

shows_per_year = (
    filtered_df["year"]
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots()
ax.plot(shows_per_year.index, shows_per_year.values)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Shows")
st.pyplot(fig)

# -----------------------------
# Section 7: Genre popularity by age group
# -----------------------------
st.header("🔥 Genre Popularity by Age Group")

df_exp = filtered_df.explode("genre_names")

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

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "**Project:** TV Shows Popularity Analysis using Synthetic Demographics  \n"
    "**Tools:** Python, Pandas, Streamlit, Matplotlib, Seaborn"
)
