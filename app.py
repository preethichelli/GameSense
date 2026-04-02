import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="GameSense", page_icon="🎮", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background-color: #0f172a;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #020617;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/steam_games_2026.csv")

# ---------------- PREPROCESS ----------------
df["Release_Date"] = pd.to_datetime(df["Release_Date"], errors="coerce")
df["Release_Year"] = df["Release_Date"].dt.year

df["combined_features"] = (
    df["Primary_Genre"].fillna('') + " " +
    df["All_Tags"].fillna('').str.replace(";", " ")
)

# ---------------- LOAD ML MODEL ----------------
model = joblib.load("models/rf_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# ---------------- TF-IDF SEARCH ----------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎛️ Filters")

genres = ["All"] + sorted(df["Primary_Genre"].dropna().unique())
selected_genre = st.sidebar.selectbox("Genre", genres)

price_range = st.sidebar.slider(
    "Price Range",
    float(df["Price_USD"].min()),
    float(df["Price_USD"].max()),
    (0.0, float(df["Price_USD"].max()))
)

year_range = st.sidebar.slider(
    "Release Year",
    int(df["Release_Year"].min()),
    int(df["Release_Year"].max()),
    (int(df["Release_Year"].min()), int(df["Release_Year"].max()))
)

review_score = st.sidebar.slider("Min Review Score", 0, 100, 0)
min_reviews = st.sidebar.slider("Min Reviews", 0, int(df["Total_Reviews"].max()), 0)

deck_option = st.sidebar.selectbox(
    "Steam Deck",
    ["All", "Verified", "Playable", "Unsupported", "Unknown"]
)

# ---------------- FILTER ----------------
filtered_df = df[
    (df["Price_USD"] >= price_range[0]) &
    (df["Price_USD"] <= price_range[1]) &
    (df["Release_Year"] >= year_range[0]) &
    (df["Release_Year"] <= year_range[1]) &
    (df["Review_Score_Pct"] >= review_score) &
    (df["Total_Reviews"] >= min_reviews)
]

if selected_genre != "All":
    filtered_df = filtered_df[filtered_df["Primary_Genre"] == selected_genre]

if deck_option != "All":
    filtered_df = filtered_df[filtered_df["Steam_Deck_Status"] == deck_option]

# ---------------- TITLE ----------------
st.title("SteamScope 🎮")
st.caption("Discover, analyze, and find your next game")

# ---------------- METRICS ----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Games", len(filtered_df))
col2.metric("Avg Rating", round(filtered_df["Review_Score_Pct"].mean(), 1))
col3.metric("Avg Price", round(filtered_df["Price_USD"].mean(), 2))
col4.metric("Top Genre", filtered_df["Primary_Genre"].mode()[0] if not filtered_df.empty else "N/A")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Insights", "🔍 Search", "🎮 Explore"])

# ---------------- INSIGHTS ----------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(px.histogram(filtered_df, x="Price_USD"), use_container_width=True)

    with col2:
        st.plotly_chart(px.scatter(filtered_df, x="Review_Score_Pct", y="Estimated_Owners"), use_container_width=True)

# ---------------- SEARCH ----------------
with tab2:
    st.subheader("🔍 Game Search")

    game_query = st.text_input("Search for a game")

    if game_query:
        match = df[df["Name"].str.contains(game_query, case=False, na=False)]

        if not match.empty:
            game = match.iloc[0]

            st.success(f"🎮 {game['Name']}")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rating", f"{game['Review_Score_Pct']}%")
            col2.metric("Reviews", game["Total_Reviews"])
            col3.metric("Price", f"${game['Price_USD']}")
            col4.metric("Genre", game["Primary_Genre"])

            # ---- ML Prediction ----
            def predict_game_success(game):
                # Step 1: Create full empty input with correct columns
                input_df = pd.DataFrame(columns=feature_columns)

                # Step 2: Fill values
                input_df.loc[0] = 0  # initialize row

                input_df.at[0, "Price_USD"] = game["Price_USD"]
                input_df.at[0, "Review_Score_Pct"] = game["Review_Score_Pct"]
                input_df.at[0, "Total_Reviews"] = game["Total_Reviews"]
                input_df.at[0, "24h_Peak_Players"] = game["24h_Peak_Players"]
                input_df.at[0, "Release_Year"] = game["Release_Year"]

                # Step 3: Set genre column = 1
                genre = game["Primary_Genre"]
                if genre in input_df.columns:
                    input_df.at[0, genre] = 1

                # Step 4: Predict
                prediction = model.predict(input_df)[0]
                return prediction

            prediction = predict_game_success(game)

            if prediction == 1:
                st.success("🔥 High Potential Game")
            else:
                st.warning("⚠️ Moderate Performance Game")

        else:
            st.error("🚫 Game not found")
            st.info("That game isn't in the top Steam listings yet 👀")

# ---------------- EXPLORE ----------------
with tab3:
    st.dataframe(filtered_df.head(50))

import os
#st.write(os.listdir("models"))