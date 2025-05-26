import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set path to the 'data' folder
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

# Load data
try:
    ratings = pd.read_csv(os.path.join(DATA_PATH, "rating_final.csv"))
    geoplaces = pd.read_csv(os.path.join(DATA_PATH, "geoplaces2.csv"))
    cuisine = pd.read_csv(os.path.join(DATA_PATH, "chefmozcuisine.csv"))
except FileNotFoundError as e:
    st.error(f"❌ File not found: {e}")
    st.stop()

# Merge and clean
geo_cuisine = pd.merge(geoplaces, cuisine, on="placeID", how="inner")
full_data = pd.merge(geo_cuisine, ratings, on="placeID", how="inner")
full_data.rename(columns={'Rcuisine': 'cuisine'}, inplace=True)

# Ensure required columns exist
required_columns = ['placeID', 'name', 'city', 'cuisine', 'price', 'rating']
for col in required_columns:
    if col not in full_data.columns:
        st.error(f"Missing column: {col}")
        st.stop()

# Select and clean data
full_data = full_data[required_columns]
full_data.drop_duplicates(inplace=True)
full_data.dropna(inplace=True)

# Combine features
full_data['combined'] = full_data.apply(lambda row: f"{row['cuisine']} {row['price']} {row['rating']}", axis=1)

# Vectorize combined features
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(full_data['combined'])
similarity = cosine_similarity(features)

# Get index from restaurant name
def get_index(name):
    return full_data[full_data.name == name].index[0]

# Recommendation function
def recommend(restaurant_name, num=5):
    try:
        idx = get_index(restaurant_name)
        scores = list(enumerate(similarity[idx]))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:num+1]
        results = []
        for i, score in sorted_scores:
            results.append({
                'Name': full_data.iloc[i]['name'],
                'Cuisine': full_data.iloc[i]['cuisine'],
                'Similarity Score': round(score, 2)
            })
        return results
    except:
        return []

# Streamlit UI
st.title("🍽️ Restaurant Recommendation System")

restaurant_list = sorted(full_data['name'].unique())
selected_restaurant = st.selectbox("Choose a restaurant you like:", restaurant_list)

if st.button("Recommend"):
    recommendations = recommend(selected_restaurant)
    if recommendations:
        st.subheader(f"Restaurants similar to '{selected_restaurant}':")
        st.table(recommendations)
    else:
        st.warning("No similar restaurants found.")
