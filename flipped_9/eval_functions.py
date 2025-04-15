import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import random


### 1. Accuracy Metrics + Long Tail Effect

def evaluate_rating_predictions(y_true, y_pred, threshold=3.5):
    y_pred_binary = [1 if r >= threshold else 0 for r in y_pred]
    y_true_binary = [1 if r >= threshold else 0 for r in y_true]
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'Accuracy': accuracy_score(y_true_binary, y_pred_binary)
    }


def analyze_long_tail_effect(ratings_df, prediction_column='predicted', item_column='movieId', show_plot=True):
    # Calcola il numero di rating per ogni item
    movie_counts = ratings_df[item_column].value_counts()
    
    # Imposta la soglia come il 75° percentile dei conteggi
    threshold = movie_counts.quantile(0.75)
    
    # Aggiungi una colonna 'popularity' al DataFrame in base al conteggio degli item
    ratings_df['popularity'] = ratings_df[item_column].map(
        lambda x: 'long_tail' if movie_counts[x] < threshold else 'head'
    )
    
    # Calcola l'errore assoluto per ogni riga
    ratings_df['error'] = abs(ratings_df['rating'] - ratings_df[prediction_column])
    
    # Raggruppa per la categoria 'popularity' e calcola l'errore medio
    error_by_popularity = ratings_df.groupby('popularity')['error'].mean()
    
    if show_plot:
        # Visualizza la distribuzione dei conteggi per item
        plt.figure(figsize=(10, 6))
        plt.hist(movie_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='gray')
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                    label=f"Soglia (75° percentile): {threshold:.0f}")
        plt.xlabel("Numero di rating per item")
        plt.ylabel("Frequenza")
        plt.title("Distribuzione dei rating per item (Long Tail)")
        plt.legend()
        plt.show()
        
        # Mostra anche il conteggio degli item in ciascuna categoria
        long_tail_count = (movie_counts < threshold).sum()
        head_count = (movie_counts >= threshold).sum()
        print(f"Numero di item in long tail: {long_tail_count}")
        print(f"Numero di item in head: {head_count}")
    
    return error_by_popularity



### 2. Visualization per Category and Popularity

def plot_avg_error_by_genre(ratings_df, movies_df):
    df = pd.merge(ratings_df, movies_df, on='movieId')
    df['genres'] = df['genres'].str.split('|')
    df = df.explode('genres')
    df['error'] = abs(df['rating'] - df['predicted'])
    genre_error = df.groupby('genres')['error'].mean()
    genre_error.plot(kind='bar', figsize=(12, 6), title='Average Error per Genre')
    plt.show()


def plot_avg_error_by_popularity(ratings_df):
    movie_counts = ratings_df['movieId'].value_counts()
    ratings_df['popularity_bin'] = pd.qcut(ratings_df['movieId'].map(movie_counts), q=3, labels=['Low', 'Medium', 'High'])
    ratings_df['error'] = abs(ratings_df['rating'] - ratings_df['predicted_rating'])
    pop_error = ratings_df.groupby('popularity_bin')['error'].mean()
    pop_error.plot(kind='bar', figsize=(8, 5), title='Error by Popularity Bin')
    plt.show()


### 3. Ranking Evaluation - Correlation Based

def evaluate_ranking_spearman(user_ratings, predicted_ratings):
    if len(user_ratings) < 2: return np.nan
    return spearmanr(user_ratings, predicted_ratings).correlation


### 4. Top-K Precision / Recall / F1 / PR-ROC

def evaluate_topk(y_true, y_scores, k=10):
    top_k_idx = np.argsort(y_scores)[-k:][::-1]
    y_pred = np.zeros_like(y_true)
    y_pred[top_k_idx] = 1
    return {
        'Precision@k': precision_score(y_true, y_pred),
        'Recall@k': recall_score(y_true, y_pred),
        'F1@k': f1_score(y_true, y_pred)
    }


def plot_precision_recall_roc(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


### 5. Random Recommender Benchmark

def random_recommender(movie_pool, k=10):
    return random.sample(list(movie_pool), k)


def evaluate_random_recommender(user_true_items, all_items, k=10):
    y_true = [1 if item in user_true_items else 0 for item in all_items]
    y_scores = np.random.rand(len(all_items))
    return evaluate_topk(y_true, y_scores, k=k)
