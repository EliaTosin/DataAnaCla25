#%% md
# # Content-Based Recommender System using Naive Bayes
# 
# This notebook implements two types of content-based recommendation systems using the MovieLens dataset:
# 1. User-specific recommender using Naive Bayes (user profile models)
# 2. Global recommender using Kronecker product of user/item features
# 3. Evaluation methodology for realistic recommendation performance
#%% md
# ### Load and Preprocess Data
#%%
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from collections import Counter

import random
random.seed(42)
#%%
DATA_PATH = "../../ml-latest-small"

ratings = pd.read_csv(os.path.join(DATA_PATH, "ratings.csv"))
movies = pd.read_csv(os.path.join(DATA_PATH, "movies.csv"))
tags = pd.read_csv(os.path.join(DATA_PATH, "tags.csv"))

## 🧹 Preprocess Movie Metadata
tags_agg = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
movies = movies.merge(tags_agg, on="movieId", how="left")
movies["tag"] = movies["tag"].fillna("")
movies["content"] = movies["genres"].str.replace("|", " ") + " " + movies["tag"]
#%%
movies.head()
#%%
movies_with_genres = movies.copy()
movies_with_genres['genres'] = movies_with_genres['genres'].str.split('|')

# Just for sake of visualization, we replace the class name
movies['genres'] = movies['genres'].replace('(no genres listed)', 'No_genres_listed')

# Creating a list with all genres
all_genres = '|'.join(movies['genres']).split('|')

# Counting all the genres
genre_counts = Counter(all_genres)

# Creating a DataFrame based on the counter
genre_counts_df = pd.DataFrame(genre_counts.items(), columns=['Genre', 'Count']).sort_values(by='Count', ascending=False)

# Create the bar chart
plt.figure(figsize=(12, 6))
plt.bar(genre_counts_df['Genre'], genre_counts_df['Count'], color='skyblue')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.title('Number of Movies by Genre')
plt.xticks(rotation=45, ha='right')
plt.show()
#%%
print(f"Number of unique tags: {len(tags['tag'].unique())}")

no_tags_count = len(movies[movies["tag"] == ""])
print(f"Number of films without tags: {no_tags_count}")

with_tags_count = len(movies[movies["tag"] != ""])
print(f"Number of films with tags: {with_tags_count}")
#%%
tags_count = tags["tag"].value_counts()

plt.figure(figsize=(12, 6))
plt.bar(tags_count.index[:20], tags_count.values[:20], color='skyblue')
plt.xticks(rotation=90)
plt.xlabel("Top 20 Tags")
plt.ylabel("Count")
plt.title("Distribution of Top 20 Tags")
plt.show()
#%% md
# ## 1. User-Specific Naive Bayes Recommender
#%% md
# The model is trained on metadata including the movie title and genres, with titles cleaned to remove release years.
#%%
# Clean the title by removing the year in parentheses
def clean_title(title):
    return re.sub(r'\s*\(\d{4}\)', '', title)
#%%
# metadata available for each movie
metadata = movies[["movieId", "title", "genres"]]
metadata.head()
#%% md
# Removing the year from movie titles helps clean the data for content-based recommendation. The year adds no semantic value for models using text features like TF-IDF and can introduce noise or inflate the vocabulary. By focusing on the actual title, we ensure better feature extraction and more accurate similarity comparisons between movies.
#%%
metadata.loc[:, 'title'] = metadata['title'].apply(clean_title)
#%%
metadata.head()
#%% md
# ### Binarize the genres
# 
# Binarizing genres turns categorical data into a binary format, making it easier for machine learning models to process. This method helps handle movies with multiple genres and captures interactions between them, improving recommendation accuracy. It simplifies the feature engineering process and ensures the model can effectively learn from genre information.
#%%
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(metadata['genres'].str.split('|'))
#%%
# Create a DataFrame with the encoded genres
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
genres_df.head()
#%%
# Concatenate the original metadata with the encoded genres
metadata = pd.concat([metadata[['movieId', 'title']], genres_df], axis=1)
# metadata = metadata.drop(columns=['(no genres listed)'])
metadata.head()
#%%
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')

# Create a preprocessor that transforms the movie metadata:
# - Applies TF-IDF vectorization to the cleaned 'title' column to extract textual features.
# - Passes through the binary genre columns (already transformed by MultiLabelBinarizer).
# - Drops any remaining columns that are not explicitly selected.
preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', tfidf, 'title'),
        ('genres', 'passthrough', genres_df.columns)
    ],
    remainder='drop'
)
#%%
def predict_single_movie(user_id, movie_id):
    # Step 1: Prepare user data
    user_ratings = ratings[ratings['userId'] == user_id]
    user_data = pd.merge(user_ratings, metadata, on='movieId')

    # Step 2: Create binary labels
    user_data['label'] = user_data['rating'].apply(lambda r: 1 if r >= 4 else (0 if r <= 2 else None))
    user_data = user_data.dropna(subset=['label'])
    user_data['label'] = user_data['label'].astype(int)

    if user_data.empty:
        print("User has insufficient data.")
        return None

    # Step 3: Train the model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MultinomialNB())
    ])

    X_train = user_data.drop(columns=['userId', 'rating', 'label', 'timestamp'])
    y_train = user_data['label']

    display(X_train.head())

    pipeline.fit(X_train, y_train)

    # Step 4: Check if the movie has already been watched
    if movie_id in user_ratings['movieId'].values:
        print("The movie has already been watched by the user.")
        return None

    # Step 5: Extract features of the requested movie
    movie_row = metadata[metadata['movieId'] == movie_id]
    display(movie_row)

    if movie_row.empty:
        print("Movie ID not found in the metadata.")
        return None

    input_cols = list(X_train.columns)
    movie_features = movie_row[input_cols]

    # Step 6: Predict the probability of liking the movie
    probs = pipeline.predict_proba(movie_features)[0]  # P(liked | features)
    if len(probs) < 2:
        print(f"The model for user {user_id} has seen only one class (not like/like).")
        return None
    else:
        prob = probs[1]

    title = movie_row['title'].values[0]

    return {
        'movieId': movie_id,
        'title': title,
        'score': prob,
        'recommended': prob >= 0.5
    }
#%%
chosen_user = random.choice(ratings['userId'].unique())
chosen_film = random.choice(metadata['movieId'].unique())

predict_single_movie(user_id=chosen_user, movie_id=chosen_film)
#%% md
# ## 2. Global Content-Based Recommender (Single Model for All Users)
#%%
metadata_with_tags = metadata.merge(movies[["movieId", "tag"]], on='movieId', how='left')
metadata_with_tags.head()
#%%
preprocessor_global = ColumnTransformer(
    transformers=[
        ('tfidf', tfidf, 'title'),
        ('genres', 'passthrough', genres_df.columns),
        ('tfidf_tag', tfidf, 'tag')
    ],
    remainder='drop'
)
#%%
def train_global_model():
    # Step 1: Prepare data
    data = pd.merge(ratings, metadata_with_tags, on='movieId')

    # Step 2: Create binary labels
    data['label'] = data['rating'].apply(lambda r: 1 if r >= 4 else (0 if r <= 2 else None))
    data = data.dropna(subset=['label'])
    data['label'] = data['label'].astype(int)

    # Step 3: Train the model
    pipeline = Pipeline([
        ('preprocessor', preprocessor_global),
        ('classifier', MultinomialNB())
    ])

    X = data.drop(columns=['userId', 'movieId', 'rating', 'label', 'timestamp'])
    y = data['label']

    display(X.head())

    model = pipeline.fit(X, y)
    return model, X.columns

model_global, train_columns = train_global_model()

def recommend_global(user_id, movie_id):
    # Step 4: Check if the movie has already been watched
    user_ratings = ratings[ratings['userId'] == user_id]
    if movie_id in user_ratings['movieId'].values:
        print("The movie has already been watched by the user.")
        return None

    # Step 5: Extract features of the requested movie
    movie_row = metadata_with_tags[metadata_with_tags['movieId'] == movie_id]
    display(movie_row)

    if movie_row.empty:
        print("Movie ID not found in the metadata.")
        return None
    movie_features = movie_row[train_columns]

    # Step 6: Predict the probability of liking the movie
    probs = model_global.predict_proba(movie_features)[0]  # P(liked | features)
    if len(probs) < 2:
        print(f"The model for user {user_id} has seen only one class (not like/like).")
        return None
    else:
        prob = probs[1]

    title = movie_row['title'].values[0]

    return {
        'movieId': movie_id,
        'title': title,
        'score': prob,
        'recommended': prob >= 0.5
    }
#%%
recommend_global(user_id=chosen_user, movie_id=chosen_film)