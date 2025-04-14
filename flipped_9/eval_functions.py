import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, \
    f1_score, roc_curve, precision_recall_curve, confusion_matrix


### 1. Accuracy Metrics + Long Tail Effect

def evaluate_rating_predictions(y_true, y_pred, threshold=3.5):
    y_pred_binary = [1 if r >= threshold else 0 for r in y_pred]
    y_true_binary = [1 if r >= threshold else 0 for r in y_true]
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'Accuracy': accuracy_score(y_true_binary, y_pred_binary)
    }


def analyze_long_tail_effect(ratings_df, prediction_column='predicted_rating'):
    movie_counts = ratings_df['movieId'].value_counts()
    threshold = movie_counts.quantile(0.75)
    ratings_df['popularity'] = ratings_df['movieId'].map(
        lambda x: 'long_tail' if movie_counts[x] < threshold else 'head')
    ratings_df['error'] = abs(ratings_df['rating'] - ratings_df[prediction_column])
    return ratings_df.groupby('popularity')['error'].mean()


def plot_confusion_matrix(y_true, y_pred, threshold=3.5):
    y_pred_binary = [1 if r >= threshold else 0 for r in y_pred]
    y_true_binary = [1 if r >= threshold else 0 for r in y_true]
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


### 2. Visualization per Category and Popularity

def plot_avg_error_by_genre(ratings_df, movies_df, prediction_column='predicted_rating'):
    df = pd.merge(ratings_df, movies_df, on='movieId')
    df['genres'] = df['genres'].str.split('|')
    df = df.explode('genres')
    df['error'] = abs(df['rating'] - df[prediction_column])
    genre_error = df.groupby('genres')['error'].mean()
    genre_error.plot(kind='bar', figsize=(12, 6), title='Average Error per Genre')
    plt.show()


def plot_avg_error_by_popularity(ratings_df, prediction_column='predicted_rating'):
    movie_counts = ratings_df['movieId'].value_counts()
    # splits those counts into 3 equal-sized quantile bins
    ratings_df['popularity_bin'] = pd.qcut(ratings_df['movieId'].map(movie_counts), q=3,
                                           labels=['Low', 'Medium', 'High'])
    ratings_df['error'] = abs(ratings_df['rating'] - ratings_df[prediction_column])
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
