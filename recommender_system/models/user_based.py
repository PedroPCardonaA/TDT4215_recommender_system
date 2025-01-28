import polars as pl
import numpy as np


class CollaborativeRecommender:
    def __init__(self, impressions: pl.DataFrame, items: pl.DataFrame, scroll_percentage_weight=1, read_time_weight=1):
        '''
        Initialize the CollaborativeRecommender with a user-item dataframe.

        Parameters
        ----------
        impressions : pl.DataFrame
            A DataFrame containing user interactions with articles.
        items : pl.DataFrame
            A DataFrame containing item details (e.g., articles).
        scroll_percentage_weight : float, optional
            The weight for the scroll percentage in the impression score.
        read_time_weight : float, optional
            The weight for the read time in the impression score.
        '''
        self.impressions = impressions
        self.items = items
        self.scroll_percentage_weight = scroll_percentage_weight
        self.read_time_weight = read_time_weight
        self.user_similarity_matrix = None

    def cosine_similarity(self, user1_score: np.ndarray, user2_score: np.ndarray) -> float:
        '''
        Calculate the cosine similarity between two vectors.

        Parameters
        ----------
        user1_score : np.ndarray
            A numpy array representing the behavior of user 1.
        user2_score : np.ndarray
            A numpy array representing the behavior of user 2.

        Returns
        -------
        float
            The cosine similarity score between the two vectors. Ranges from -1 to 1.
        '''
        norm_u = np.linalg.norm(user1_score)
        norm_v = np.linalg.norm(user2_score)

        # Handle division by zero
        if norm_u == 0 or norm_v == 0:
            return 0.0

        return np.dot(user1_score, user2_score) / (norm_u * norm_v)

    def add_impression_scores(self) -> pl.DataFrame:
        '''
        Adds an impression score column to the `impressions` DataFrame.

        Returns
        -------
        pl.DataFrame
            A DataFrame with an additional column `impression_score`.
        '''
        self.impressions = self.impressions.with_columns(
            (
                pl.col("scroll_percentage") * self.scroll_percentage_weight +
                pl.col("read_time") * self.read_time_weight
            ).alias("impression_score")
        )
        return self.impressions

    def build_user_similarity_matrix(self):
        '''
        Builds a user-user similarity matrix using cosine similarity based on impression scores.

        The matrix is stored as a dictionary of dictionaries where keys are user IDs
        and values are their similarity scores with other users.
        '''
        # Pivot the data to create a user-item matrix
        user_item_matrix = self.impressions.pivot(
            values="impression_score",
            index="user_id",
            columns="article_id",
            aggregate_function="mean"
        ).fill_nan(0)

        # Get user IDs and scores as numpy arrays
        user_ids = user_item_matrix["user_id"].to_numpy()
        scores_matrix = user_item_matrix.drop("user_id").to_numpy()

        # Calculate similarity matrix
        norm_matrix = np.linalg.norm(scores_matrix, axis=1, keepdims=True)
        norm_matrix[norm_matrix == 0] = 1  # Prevent division by zero
        normalized_matrix = scores_matrix / norm_matrix

        similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)

        # Convert to dictionary for faster lookups
        self.user_similarity_matrix = {
            user_ids[i]: {user_ids[j]: similarity_matrix[i, j] for j in range(len(user_ids))}
            for i in range(len(user_ids))
        }
        return self.user_similarity_matrix

    def get_n_similar_users(self, user_id: int, n: int) -> list:
        '''
        Finds the n most similar users to a given user based on the similarity matrix.

        Parameters
        ----------
        user_id : int
            The ID of the user for whom similar users are being found.
        n : int
            The number of similar users to return.

        Returns
        -------
        list
            A list of tuples where each tuple contains a user ID and its similarity score.
        '''
        if self.user_similarity_matrix is None:
            raise ValueError("User similarity matrix has not been computed. Call `fit` first.")

        similarities = self.user_similarity_matrix.get(user_id, {})

        # Sort by similarity score in descending order and return the top n, excluding the user itself
        similar_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [(uid, score) for uid, score in similar_users if uid != user_id][:n]

    def fit(self):
        '''
        Fits the Collaborative Recommender model by building the user similarity matrix.

        Returns
        -------
        dict
            The user-user similarity matrix.
        '''
        self.add_impression_scores()
        return self.build_user_similarity_matrix()

    def recommend_n_articles(self, user_id: int, n: int) -> list[int]:
        '''
        Predict the top n articles a user might like based on similar users activity.

        Parameters
        ----------
        user_id : int
            The ID of the user for whom to make predictions.
        n : int
            The number of articles to recommend.

        Returns
        -------
        list[int]
            A list of article IDs predicted to be most liked by the user.
        '''
        # Get the n most similar users
        similar_users = [uid for uid, _ in self.get_n_similar_users(user_id, n)]

        # Get articles interacted with by similar users
        similar_user_articles = self.impressions.filter(
            pl.col("user_id").is_in(similar_users)
        )

        # Aggregate scores for each article
        article_scores = similar_user_articles.groupby("article_id").agg(
            pl.col("impression_score").sum().alias("total_score")
        )

        # Sort by scores in descending order and take the top n articles
        recommended_articles = article_scores.sort("total_score", descending=True).head(n)

        return recommended_articles["article_id"].to_list()
