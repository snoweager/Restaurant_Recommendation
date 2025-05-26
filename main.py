import pandas as pd

# Load datasets
geoplaces = pd.read_csv("data/geoplaces2.csv", encoding="latin-1")
cuisine = pd.read_csv("data/chefmozcuisine.csv", encoding="latin-1")
ratings = pd.read_csv("data/rating_final.csv", encoding="latin-1")

# Preview
print("Geoplaces:\n", geoplaces.head())
print("Cuisine:\n", cuisine.head())
print("Ratings:\n", ratings.head())

# Merge geoplaces with cuisine info
geo_cuisine = pd.merge(geoplaces, cuisine, on="placeID", how="inner")

# Merge with ratings
full_data = pd.merge(geo_cuisine, ratings, on="placeID", how="inner")

# Rename Rcuisine to cuisine for consistency
full_data.rename(columns={'Rcuisine': 'cuisine'}, inplace=True)

# Select only necessary columns
full_data = full_data[['placeID', 'name', 'city', 'cuisine', 'price', 'rating']]

# Clean
full_data.drop_duplicates(inplace=True)
full_data.dropna(inplace=True)

# Show result
print("\nCleaned Combined Data:\n", full_data.head())

# Create a new combined feature column
def combine_features(row):
    return f"{row['cuisine']} {row['price']} {row['rating']}"

# Apply to each row
full_data['combined'] = full_data.apply(combine_features, axis=1)

# Show example
print("\nCombined Feature Text:\n", full_data[['name', 'combined']].head())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Vectorize the 'combined' column
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(full_data['combined'])

# Calculate cosine similarity between all restaurants
similarity = cosine_similarity(features)

# Helper: Get restaurant index by name
def get_index(name):
    return full_data[full_data.name == name].index[0]

# Recommendation function
def recommend(restaurant_name, num=5):
    try:
        idx = get_index(restaurant_name)
        scores = list(enumerate(similarity[idx]))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:num+1]

        print(f"\nRestaurants similar to '{restaurant_name}':\n")
        for i, score in sorted_scores:
            print(f"{full_data.iloc[i]['name']} | Cuisine: {full_data.iloc[i]['cuisine']} | Score: {round(score, 2)}")

    except IndexError:
        print("Restaurant not found. Please check the spelling or pick a different one.")

# Show all restaurant names so the user can choose one
print("\nAvailable restaurant names:")
print(full_data['name'].unique()[:20])  # Show first 20 for reference

# Test the function
recommend("Kiku Cuernavaca")

print("\n--- Test 1 ---")
recommend("puesto de tacos")

print("\n--- Test 2 ---")
recommend("little pizza Emilio Portes Gil")

print("\n--- Test 3 ---")
recommend("Sirlone")  # if it's in your dataset

