#%%
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
#%%
import random
random.seed(42)
#%%
DATA_PATH = "../../ml-latest-small"

ratings = pd.read_csv(os.path.join(DATA_PATH, "ratings.csv"))
movies = pd.read_csv(os.path.join(DATA_PATH, "movies.csv"))
tags = pd.read_csv(os.path.join(DATA_PATH, "tags.csv"))
#%% md
# # Naive Bayes CF Model Based (Week6)
#%%
df_ecommerce= pd.read_json('1_ecommerce.jsonl', lines=True)

# for each session, we will create a list of items that the user has clicked on, removing duplicates
clicks_items_list = []
carts_items_list = []
orders_items_list = []

for events in df_ecommerce.events:
    clicks = []
    carts = []
    orders = []
    for e in events:
        if e['type'] == 'clicks':
            clicks.append(e['aid'])
        if e['type'] == 'carts':
            carts.append(e['aid'])
        if e['type'] == 'orders':
            orders.append(e['aid'])

    clicks_items_list.append(list(clicks))
    carts_items_list.append(list(carts))
    orders_items_list.append(list(orders))

df_ecommerce['clicks'] = clicks_items_list
df_ecommerce['carts'] = carts_items_list
df_ecommerce['orders'] = orders_items_list
df = df_ecommerce.drop(columns=["events"], axis=1)

clicks_length = df_ecommerce['clicks'].apply(len)
carts_length = df_ecommerce['carts'].apply(len)
orders_length = df_ecommerce['orders'].apply(len)

# takes only the session with items length higher than 20
df_truncated = df[df['clicks'].apply(lambda x: len(x) > 10)]
df = df_truncated.copy()
# redefine index
df.reset_index(drop=True, inplace=True)

# Explode each column (clicks, carts, orders)
df_clicks = df[['session', 'clicks']].explode('clicks').rename(columns={'clicks': 'item'}).dropna(subset=['item'])
df_carts = df[['session', 'carts']].explode('carts').rename(columns={'carts': 'item'}).dropna(subset=['item'])
df_orders = df[['session', 'orders']].explode('orders').rename(columns={'orders': 'item'}).dropna(subset=['item'])

# Concatenate the exploded dataframes
df_concat = pd.concat([df_clicks, df_carts, df_orders])

# Create a new column for each category indicating whether the item is present in that category
df_concat['click'] = df_concat['item'].isin(df_clicks['item']).astype(int)
df_concat['cart'] = df_concat['item'].isin(df_carts['item']).astype(int)
df_concat['order'] = df_concat['item'].isin(df_orders['item']).astype(int)

# Drop duplicates based on session and item
df_concat = df_concat.drop_duplicates(subset=['session', 'item'])
#%%
def sgd_matrix_factorization(df, k=10, alpha=0.01, lambda_reg=0.1, num_epochs=20, w_click=1, w_cart=3, w_order=5, test_size=0.2, validation_size=0.1):
    # Map session (users) and items to consecutive indices
    users = {u: i for i, u in enumerate(df['session'].unique())}
    items = {i: j for j, i in enumerate(df['item'].unique())}

    num_users = len(users)
    num_items = len(items)

    # Initialize U, V, and biases
    U = np.random.rand(num_users, k)
    V = np.random.rand(num_items, k)
    b_u = np.zeros(num_users)
    b_i = np.zeros(num_items)
    b = 0  # Global bias

    # Create (u, i, r_ui) tuples for all interactions
    data = []

    # iter throw all the rows of the dataframe
    # and create a list of tuples (user, item, rating)
    # Assign ratings to interactions
    for _, row in df.iterrows():
        u = users[row['session']]
        i = items[row['item']]

        # if this user has perform an action on this item, we assign a rating based on the action
        # type starting from the most important one
        # (order > cart > click)
        if row['order'] > 0:
            r_ui = w_order
        elif row['cart'] > 0:
            r_ui = w_cart
        elif row['click'] > 0:
            r_ui = w_click
        else:
            continue  # Skip interactions with no recorded action

        data.append((u, i, r_ui))

    # Split the data into training, validation, and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=validation_size, random_state=42)
#%% md
# # Unconstrained Matrix Factorisation based Collaborative Filtering (week 7)
#%%
df_ratings = pd.read_csv("../ml-latest-small/ratings.csv")

df_ratings.loc[df_ratings['rating'] <= 2, 'rating_ordinal'] = "Don't like"
df_ratings.loc[(df_ratings['rating'] > 2) & (df_ratings['rating'] <= 4), 'rating_ordinal'] = "Like"
df_ratings.loc[df_ratings['rating'] > 4, 'rating_ordinal'] = "Really like"

ratings_ordinals = df_ratings['rating_ordinal'].unique()

df_movies = pd.read_csv("../ml-latest-small/movies.csv")

df_user_movie_ratings = df_ratings.pivot(
    index='userId',
    columns='movieId',
    values='rating_ordinal'
)
df_user_movie_ratings.head()

all_films = df_ratings["movieId"].unique()

# Crea una Serie booleana: True dove il rating è mancante
missing = df_user_movie_ratings.isnull()

# Trasforma in formato "long" (una riga per ogni cella)
missing = missing.stack()

# Filtra solo le celle in cui il rating è mancante
missing = missing[missing].reset_index()
missing.columns = ['userId', 'movieId', 'is_missing']

# Aggiungi la colonna 'flag' con valore False
df_ratings_missing = missing[['userId', 'movieId']].copy()
#%%
# itera per tutte le valutazioni manacanti
for row in df_ratings_missing.itertuples():
    userId = row.userId
    movieId = row.movieId

    # film seen from the user
    films_seen = df_user_movie_ratings.loc[userId].dropna().index

    for category in ratings_ordinals:
        # P(r31 = 1)

        df_current_movie = df_ratings[(df_ratings['movieId'] == movieId)]
        df_current_movie_category = df_current_movie[(df_current_movie['rating_ordinal'] == category)]
        users_that_voted_current_movies_with_this_category = df_current_movie_category['userId'].unique()

        p_r31 = len(df_current_movie_category) / len(df_current_movie)

        probs = [p_r31]
        for film_seen in films_seen:
            # P(r32 = 1 | r31 = 1)
            # prendo la valutazione che l'utente ha dato al film visto
            assigned_rating = df_user_movie_ratings.loc[userId, film_seen]

            # cerco tutti gli utenti che hanno votato il film visto come l'utente corrente è che hanno valutato il
            # film corrente con la stessa categoria

            df_seen_movie = df_ratings[(df_ratings['movieId'] == film_seen)]
            df_seen_movie_category = df_seen_movie[df_seen_movie['userId'].isin(users_that_voted_current_movies_with_this_category)]

            # di quelli, cerco quanti hanno votato il film che l'utente ha visto con la stessa categoria
            df_seen_movie_category = df_seen_movie_category[df_seen_movie_category['rating_ordinal'] == assigned_rating]

            # if no one has voted the movie with the same category, we skip it
            if len(df_seen_movie_category) == 0:
                continue

            probs.append(len(df_seen_movie_category) / len(df_seen_movie))

        # P(r32 = 1 | r31 = 1) * P(r31 = 1)
        prob = np.prod(probs)
        df_ratings_missing.loc[(df_ratings_missing['userId'] == userId) & (df_ratings_missing['movieId'] == movieId), category] = prob
#%%
from tqdm.notebook import tqdm
tqdm.pandas()

def compute_missing_prob(row):
    userId = row['userId']
    movieId = row['movieId']
    # Film che l'utente ha visto
    films_seen = df_user_movie_ratings.loc[userId].dropna().index.tolist()

    # Dizionario che conterrà il risultato per ogni categoria
    result = {}

    # Per ciascuna categoria (rating) da considerare
    for category in ratings_ordinals:
        # Calcola P(r(movieId) = category)
        df_current_movie = df_ratings[df_ratings['movieId'] == movieId]
        df_current_movie_category = df_current_movie[df_current_movie['rating_ordinal'] == category]
        # Gestione di eventuale divisione per zero:
        if len(df_current_movie) == 0:
            p_r31 = 0
        else:
            p_r31 = len(df_current_movie_category) / len(df_current_movie)

        # Lista delle probabilità da moltiplicare
        probs = [p_r31]

        # Itera sui film che l'utente ha visto
        for film_seen in films_seen:
            assigned_rating = df_user_movie_ratings.loc[userId, film_seen]
            # Filtra i voti del film visto
            df_seen_movie = df_ratings[df_ratings['movieId'] == film_seen]
            # Limita agli utenti che hanno votato il film mancante con 'category'
            users_voted_current = df_current_movie_category['userId'].unique()
            df_seen_movie_category = df_seen_movie[df_seen_movie['userId'].isin(users_voted_current)]
            # Filtra in base al rating assegnato dall'utente al film visto
            df_seen_movie_category = df_seen_movie_category[df_seen_movie_category['rating_ordinal'] == assigned_rating]

            # Se non ci sono voti, puoi decidere se saltare il film oppure applicare uno smoothing (qui si salta)
            if len(df_seen_movie) == 0:
                p_cond = 1  # oppure 0 oppure applicare smoothing
            else:
                if len(df_seen_movie_category) == 0:
                    continue
                p_cond = len(df_seen_movie_category) / len(df_seen_movie)
            probs.append(p_cond)

        # Il prodotto delle probabilità
        result[category] = np.prod(probs)

    return pd.Series(result)

# Applica la funzione a df_ratings_missing
df_ratings_missing[ratings_ordinals] = df_ratings_missing.progress_apply(compute_missing_prob, axis=1)
#%% md
# # Bayes Classification Content-based (week 8)
#%%
## 🧹 Preprocess Movie Metadata
tags_agg = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
movies = movies.merge(tags_agg, on="movieId", how="left")
movies["tag"] = movies["tag"].fillna("")
movies["content"] = movies["genres"].str.replace("|", " ") + " " + movies["tag"]
#%% md
# #### 1. User-Specific Naive Bayes Recommender
#%%
# Clean the title by removing the year in parentheses
def clean_title(title):
    return re.sub(r'\s*\(\d{4}\)', '', title)

metadata = movies[["movieId", "title", "genres"]]
metadata.loc[:, 'title'] = metadata['title'].apply(clean_title)

mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(metadata['genres'].str.split('|'))
# Create a DataFrame with the encoded genres
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
# Concatenate the original metadata with the encoded genres
metadata = pd.concat([metadata[['movieId', 'title']], genres_df], axis=1)

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
# #### 2. Global Content-Based Recommender (Single Model for All Users)
#%%
metadata_with_tags = metadata.merge(movies[["movieId", "tag"]], on='movieId', how='left')

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
#%% md
# # Evaluation
#%%
