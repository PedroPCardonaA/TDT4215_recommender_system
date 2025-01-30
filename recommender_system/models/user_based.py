import polars as pl
import numpy as np
from scipy.spatial.distance import pdist, squareform

class CollaborativeRecommender:
    def __init__(self, impressions: pl.DataFrame, scroll_percentage_weight=1, read_time_weight=1):
        '''
        Initialize the CollaborativeRecommender with a user-item dataframe.

        Parameters
        ----------
        impressions : pl.DataFrame
            A DataFrame containing user interactions with articles.
        scroll_percentage_weight : float, optional
            The weight for the scroll percentage in the impression score.
        read_time_weight : float, optional
            The weight for the read time in the impression score.
        '''
        self.impressions = impressions
        self.scroll_percentage_weight = scroll_percentage_weight
        self.read_time_weight = read_time_weight
        self.user_similarity_matrix = {}

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
                pl.col("max_scroll") * self.scroll_percentage_weight +
                pl.col("total_readtime") * self.read_time_weight
            ).alias("impression_score")
        )
        return self.impressions

    def build_user_similarity_matrix(self, sim_size=10):
        '''
        Builds a user similarity matrix using cosine similarity based on impression scores.
        Each user contains the `sim_size` most similar users, sorted by similarity.

        The matrix is stored as a dictionary of lists where the keys are user IDs
        and the values in the lists are `sim_size` instances of the most similar users, sorted by similarity.
        '''
        # Pivot to create user-item matrix
        user_item_matrix = self.impressions.pivot(
            values="impression_score",
            index="user_id",
            columns="article_id"
        ).fill_null(0)

        user_ids = user_item_matrix["user_id"].to_list()
        user_vectors = user_item_matrix.drop("user_id").to_numpy()

        # Vectorized cosine similarity calculation
        similarity_matrix = 1 - squareform(pdist(user_vectors, metric='cosine'))

        # Store top `sim_size` most similar users for each user
        top_similarities = np.argsort(-similarity_matrix, axis=1)[:, 1:sim_size+1]
        self.user_similarity_matrix = {
            user_ids[i]: [(user_ids[j], similarity_matrix[i, j]) for j in top_similarities[i]]
            for i in range(len(user_ids))
        }

        return self.user_similarity_matrix

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
        Predict the top n articles a user might like based on similar users' activity.

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
        if user_id not in self.user_similarity_matrix:
            return []  # Return empty list if user not found

        # Get the n most similar users
        similar_users = [uid for uid, _ in self.user_similarity_matrix[user_id]]

        # Get articles interacted with by similar users
        similar_user_articles = self.impressions.filter(
            pl.col("user_id").is_in(similar_users)
        )

        # Aggregate scores for each article
        article_scores = similar_user_articles.group_by("article_id").agg(
            pl.col("impression_score").sum().alias("total_score")
        )

        # Sort by scores in descending order and take the top n articles
        recommended_articles = article_scores.sort("total_score", descending=True).head(n)

        return recommended_articles["article_id"].to_list()
