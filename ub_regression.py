import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


class UserBasedRegressionModel:
    def __init__(self, k=50, regularization=0.1):
        """
        Initialize the User-Based Nearest Neighbor Regression model.

        This model extends traditional user-based collaborative filtering by learning
        regression weights for each neighbor, rather than using simple similarity weights.

        Parameters:
        -----------
        k : int
            Number of nearest neighbors to consider for each user
        regularization : float
            Regularization parameter lambda to prevent overfitting when learning weights
        """
        self.k = k  # Number of neighbors to consider
        self.regularization = regularization  # Regularization strength for weight optimization
        # Will store the mean rating for each user
        self.user_means = None
        # Will store similarity values between all pairs of users
        self.similarity_matrix = None
        # Will store optimized weights for each user's neighbors
        self.user_weights = None
        # Will store the sparse matrix of user-item ratings
        self.ratings_matrix = None
        # Will store the k nearest neighbors for each user
        self.user_neighbors = None
        # Will store lists of user IDs and item IDs
        self.users = None
        self.items = None

    def fit(self, ratings_df):
        """
        Fit the model using a user-item matrix where:
        - rows are users (userId as index)
        - columns are items (movieId as columns)
        - values are ratings
        - NaN represents missing ratings

        The fitting process involves:
        1. Converting the data to a sparse matrix format
        2. Computing mean ratings for each user
        3. Calculating similarity between users
        4. Finding k nearest neighbors for each user
        5. Learning regression weights through optimization

        Parameters:
        -----------
        ratings_df : pandas DataFrame
            DataFrame with users as rows and items as columns
        """
        # Store user and item lists for later reference and lookups
        self.users = ratings_df.index.tolist()
        self.items = ratings_df.columns.tolist()

        # Convert to sparse matrix for more efficient computation
        # First, reshape the dataframe to long format (userId, movieId, rating)
        ratings_long = ratings_df.reset_index().melt(
            id_vars='userId',
            var_name='movieId',
            value_name='rating'
        ).dropna()  # Drop NaN values (missing ratings)

        print("Sparse Matrix: ", ratings_long)

        # Create mappings from user/item IDs to matrix indices
        user_indices = {user: i for i, user in enumerate(self.users)}
        item_indices = {item: i for i, item in enumerate(self.items)}

        # Create the data needed for sparse matrix construction:
        # - rows: indices of users in the matrix
        # - cols: indices of items in the matrix
        # - data: the actual rating values
        rows = [user_indices[user] for user in ratings_long['userId']]
        cols = [item_indices[movie] for movie in ratings_long['movieId']]
        data = ratings_long['rating'].values

        # Create Compressed Sparse Row matrix - efficient for row operations
        # which is important since we'll be accessing user ratings frequently
        self.ratings_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.users), len(self.items))
        )

        # Calculate the mean rating for each user
        # This will be used for mean-centering when making predictions
        self.user_means = np.zeros(len(self.users))
        for u in range(len(self.users)):
            user_ratings = self.ratings_matrix[u].data
            if len(user_ratings) > 0:
                # If user has ratings, calculate their mean
                self.user_means[u] = user_ratings.mean()
            else:
                # If user has no ratings, use global mean
                self.user_means[u] = np.nanmean(ratings_df.values.flatten())

        # Compute similarity between all pairs of users
        # Using cosine similarity as a measure of how similar users' rating patterns are
        self._compute_pearson_similarity()

        # Determine k nearest neighbors for each user based on similarity
        self._find_user_neighbors()

        # Learn regression weights through optimization
        # This is a key step that differentiates this model from standard user-based CF
        self._learn_regression_weights()

    def _compute_pearson_similarity(self):
        """
        Calculate the similarity between users using cosine similarity.

        Despite the method name referring to Pearson correlation, the implementation
        uses cosine similarity from sklearn, which is computationally efficient
        when working with sparse matrices.

        Cosine similarity measures the cosine of the angle between two vectors,
        representing how similar their rating patterns are, regardless of magnitude.
        """
        # Compute similarity matrix where each entry [i,j] represents
        # the similarity between user i and user j
        self.similarity_matrix = cosine_similarity(self.ratings_matrix)

    def _find_user_neighbors(self):
        """
        Determine the k nearest neighbors for each user based on similarity.

        This method identifies the most similar users to each user, which will
        be used for generating predictions.
        """
        n_users = len(self.users)
        self.user_neighbors = {}

        for u in range(n_users):
            # Get indices of users sorted by decreasing similarity to user u
            similar_users = np.argsort(self.similarity_matrix[u])[::-1]

            # Remove the user itself from its neighbors list
            similar_users = similar_users[similar_users != u]

            # Take only the top k most similar users
            self.user_neighbors[u] = similar_users[:self.k]

    def _objective_function(self, weights, u, neighbors_map):
        """
        Objective function for weight optimization.

        This function calculates the sum of squared errors between actual ratings
        and predicted ratings, plus a regularization term. The goal of optimization
        is to minimize this value.

        Parameters:
        -----------
        weights : numpy array
            Flattened array of weights to be optimized
        u : int
            User index for whom we're optimizing weights
        neighbors_map : dict
            Mapping from neighbor indices to positions in the weights array

        Returns:
        --------
        float : Value of the objective function (error + regularization)
        """
        # Reconstruct weights dictionary from flattened array for easier indexing
        user_weights = {v: weights[i] for v, i in neighbors_map.items()}

        # Get indices of items that user u has rated
        items_rated_by_u = self.ratings_matrix[u].indices

        # Initialize sum of squared errors
        squared_errors = 0

        # Calculate error for each item rated by user u
        for j in items_rated_by_u:
            # Find neighbors who have also rated item j
            neighbors_who_rated_j = [v for v in self.user_neighbors[u]
                                     if j in self.ratings_matrix[v].indices]

            # This is the set P_u(j) as defined in the paper:
            # the set of neighbors of u who have rated item j
            Pu_j = neighbors_who_rated_j

            # Calculate prediction using mean-centered approach
            # Start with user u's mean rating
            prediction = self.user_means[u]

            # Add weighted deviations from neighbors' means
            for v in Pu_j:
                if v in user_weights:
                    # Add the weighted deviation of neighbor v's rating from their mean
                    prediction += user_weights[v] * (self.ratings_matrix[v, j] - self.user_means[v])

            # Get actual rating that user u gave to item j
            actual_rating = self.ratings_matrix[u, j]

            # Add squared error to total
            squared_errors += (actual_rating - prediction) ** 2

        # Add regularization term (L2 regularization) to prevent overfitting
        # This penalizes large weight values
        regularization_term = self.regularization * np.sum(weights ** 2)

        # Return total objective function value
        return squared_errors + regularization_term

    def _learn_regression_weights(self):
        """
        Learn regression weights through optimization for each user.

        This method optimizes the weights for each user's neighbors to minimize
        the prediction error on the items the user has already rated.
        These weights are then used instead of simple similarity values
        when making predictions.
        """
        n_users = len(self.users)
        # Initialize weights dictionary for all users
        self.user_weights = {u: {} for u in range(n_users)}

        # Optimize weights for each user
        for u in range(n_users):
            # Skip users with no ratings (can't learn meaningful weights)
            if len(self.ratings_matrix[u].indices) == 0:
                continue

            # Get neighbors for user u
            neighbors = self.user_neighbors[u]

            # Skip if no neighbors found
            if len(neighbors) == 0:
                continue

            # Create mapping from neighbor index to position in weights array
            # This is needed for the optimization function
            neighbors_map = {v: i for i, v in enumerate(neighbors)}

            # Initialize weights with similarity values as a starting point
            initial_weights = np.array([self.similarity_matrix[u, v] for v in neighbors])

            try:
                # Optimize weights using L-BFGS-B algorithm
                # This minimizes the objective function with respect to the weights
                result = minimize(
                    self._objective_function,  # Function to minimize
                    initial_weights,  # Initial values for weights
                    args=(u, neighbors_map),  # Additional arguments to the function
                    method='L-BFGS-B',  # Optimization algorithm
                    options={'disp': False, 'maxiter': 100}  # Options
                )

                # Store optimized weights in the user_weights dictionary
                optimized_weights = result.x
                for i, v in enumerate(neighbors):
                    self.user_weights[u][v] = optimized_weights[i]

            except Exception as e:
                # If optimization fails, fall back to similarity weights
                print(f"Optimization failed for user {self.users[u]}: {str(e)}")
                for i, v in enumerate(neighbors):
                    self.user_weights[u][v] = self.similarity_matrix[u, v]

    def predict(self, user_id, movie_id):
        """
        Predict rating for a specific user and movie.

        This uses the learned regression weights to predict how a user would
        rate a movie they haven't seen yet, based on ratings from similar users.

        Parameters:
        -----------
        user_id : user identifier from the original dataframe index
        movie_id : movie identifier from the original dataframe columns

        Returns:
        --------
        float : predicted rating, or None if prediction cannot be made
        """
        # Check if user and movie exist in the model's training data
        if user_id not in self.users or movie_id not in self.items:
            return None

        # Convert external IDs to internal indices
        u = self.users.index(user_id)
        j = self.items.index(movie_id)

        # If user has already rated this movie, return the actual rating
        if j in self.ratings_matrix[u].indices:
            return self.ratings_matrix[u, j]

        # Get neighbors of user u
        if u not in self.user_neighbors:
            # No neighbors found, return user's mean rating
            return self.user_means[u]

        neighbors = self.user_neighbors[u]

        # Find neighbors who have rated item j
        neighbors_who_rated_j = [v for v in neighbors if j in self.ratings_matrix[v].indices]

        # This is the set P_u(j) from the paper
        Pu_j = neighbors_who_rated_j

        # If no neighbors rated this item, return user's mean rating
        if len(Pu_j) == 0:
            return self.user_means[u]

        # Calculate prediction using the weighted average formula from the paper (Equation 2.22)
        # Start with the user's mean rating
        prediction = self.user_means[u]

        # Add weighted deviations of neighbors' ratings from their means
        for v in Pu_j:
            if v in self.user_weights.get(u, {}):
                # Get the optimized weight for this neighbor
                weight = self.user_weights[u][v]
                # Add weighted deviation
                prediction += weight * (self.ratings_matrix[v, j] - self.user_means[v])

        return prediction

    def recommend_movies(self, user_id, n_recommendations=10, exclude_rated=True):
        """
        Recommend top N movies for a user based on predicted ratings.

        Parameters:
        -----------
        user_id : user identifier from the original dataframe index
        n_recommendations : int, number of recommendations to return
        exclude_rated : bool, whether to exclude already rated movies

        Returns:
        --------
        list : list of (movie_id, predicted_rating) tuples, sorted by predicted rating
        """
        # Check if user exists in the model
        if user_id not in self.users:
            return []

        # Get user index
        u = self.users.index(user_id)

        # Get all movies
        all_movies = self.items

        # Exclude already rated movies if required
        if exclude_rated:
            # Get movies that user has already rated
            rated_movies_indices = self.ratings_matrix[u].indices
            rated_movies = [self.items[idx] for idx in rated_movies_indices]
            # Filter out rated movies
            movies_to_predict = [movie for movie in all_movies if movie not in rated_movies]
        else:
            # Consider all movies
            movies_to_predict = all_movies

        # Generate predictions for all candidate movies
        predictions = []
        for movie in movies_to_predict:
            # Predict rating for this movie
            pred = self.predict(user_id, movie)
            if pred is not None:
                predictions.append((movie, pred))

        # Sort by predicted rating (descending) and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

    def evaluate(self, test_df):
        """
        Evaluate the model on test data.

        Calculates Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)
        to measure prediction accuracy.

        Parameters:
        -----------
        test_df : pandas DataFrame in long format with columns 'userId', 'movieId', 'rating'

        Returns:
        --------
        dict : Dictionary with RMSE and MAE metrics
        """
        # Filter test data to include only users and items known to the model
        test_df = test_df[
            test_df['userId'].isin(self.users) &
            test_df['movieId'].isin(self.items)
            ]

        # Return None for metrics if no valid test data
        if len(test_df) == 0:
            return {'RMSE': None, 'MAE': None}

        # Lists to store predictions and actual ratings
        predictions = []
        actual_ratings = []

        # Generate predictions for each test case
        for _, row in test_df.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            rating = row['rating']

            # Get prediction
            pred = self.predict(user_id, movie_id)
            if pred is not None:
                predictions.append(pred)
                actual_ratings.append(rating)

        # Calculate metrics
        # RMSE: Root Mean Squared Error - sensitive to large errors
        rmse = np.sqrt(mean_squared_error(actual_ratings, predictions))
        # MAE: Mean Absolute Error - average absolute difference between predicted and actual
        mae = np.mean(np.abs(np.array(actual_ratings) - np.array(predictions)))

        return {'RMSE': rmse, 'MAE': mae}


# Example usage with the provided user-item matrix format
if __name__ == "__main__":
    # Load movie data from the MovieLens dataset
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    tags = pd.read_csv('ml-latest-small/tags.csv')

    # Convert ratings to user-item matrix format
    # - Each row represents a user
    # - Each column represents a movie
    # - Each cell contains the rating (or NaN if not rated)
    ratings_df = ratings.pivot(index='userId', columns='movieId', values='rating')
    ratings_df.index.name = 'userId'

    print("User-Item Matrix:")
    print(ratings_df)

    # Train the model with k=10 neighbors and regularization=0.1
    print("\nTraining User-Based Nearest Neighbor Regression model...")
    model = UserBasedRegressionModel(k=10, regularization=0.1)
    model.fit(ratings_df)

    # Get and display recommendations for user 1
    print("\nTop 5 recommendations for user 1:")
    recommendations = model.recommend_movies(1, n_recommendations=5)
    for movie_id, pred_rating in recommendations:
        print(f"Movie {movie_id}: Predicted rating {pred_rating:.2f}")

    # Get and display recommendations for user 500
    print("\nTop 5 recommendations for user 500:")
    recommendations = model.recommend_movies(500, n_recommendations=5)
    for movie_id, pred_rating in recommendations:
        print(f"Movie {movie_id}: Predicted rating {pred_rating:.2f}")

    # Convert to long format for evaluation
    ratings_long = ratings_df.reset_index().melt(
        id_vars='userId',
        var_name='movieId',
        value_name='rating'
    ).dropna()

    train_df, test_df = train_test_split(ratings_long, test_size=0.2, random_state=42)

    # Re-train on training data only
    train_matrix = train_df.pivot(index='userId', columns='movieId', values='rating')
    model = UserBasedRegressionModel(k=10, regularization=0.1)
    model.fit(train_matrix)

    # Evaluate on test data
    metrics = model.evaluate(test_df)
    print("\nModel evaluation:")
    print(f"RMSE: {metrics['RMSE']:.4f}" if metrics['RMSE'] else "RMSE: Not enough data")
    print(f"MAE: {metrics['MAE']:.4f}" if metrics['MAE'] else "MAE: Not enough data")
