import polars as pl
import numpy as np
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed

class CollaborativeRecommender:
    def __init__(self, impressions: pl.DataFrame):
        '''
        Initialize the CollaborativeRecommender with a user-item dataframe.

        Parameters
        ----------
        impressions : pl.DataFrame
            A DataFrame containing user interactions with articles.
        '''
        self.impressions = impressions
        self.user_similarity_matrix = {}

    def add_impression_scores(self, scroll_weight: float = 1.0, readtime_weight: float = 1.0) -> pl.DataFrame:
        """
        Computes and adds an `impression_score` column to the `impressions` DataFrame.

        The impression score is calculated as a weighted sum of the `max_scroll` and `total_readtime` columns:
        
            impression_score = (max_scroll * scroll_weight) + (total_readtime * readtime_weight)

        Parameters
        ----------
        scroll_weight : float, optional
            The weight assigned to the `max_scroll` column (default is 1.0).
        readtime_weight : float, optional
            The weight assigned to the `total_readtime` column (default is 1.0).

        Returns
        -------
        pl.DataFrame
            A DataFrame with an additional `impression_score` column.
        """
        self.impressions = self.impressions.with_columns(
            (
                pl.col("max_scroll") * scroll_weight +
                pl.col("total_readtime") * readtime_weight
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
        top_similarities = np.argsort(-similarity_matrix, axis=1)[:, 1:sim_size + 1]
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

    def recommend_n_articles(self, user_id: int, n: int, allow_read=False) -> list[int]:
        '''
        Predict the top n articles a user might like based on similar users' activity,
        ensuring that articles the user has already read are not recommended.

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

        # Get articles the user has already read
        user_articles = set(
            self.impressions.filter(pl.col("user_id") == user_id)["article_id"].to_list()
        )

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

        # Remove articles the user has already read
        if allow_read:
            filtered_articles = article_scores
        else:
            filtered_articles = article_scores.filter(~pl.col("article_id").is_in(user_articles))

        # Sort by scores in descending order and take the top n articles
        recommended_articles = filtered_articles.sort("total_score", descending=True).head(n)

        return recommended_articles["article_id"].to_list()

    # Accuracy functions from content based

    def precision_at_k(self, recommended_items, relevant_items, k=5):
        """
        Compute Precision@K.
        
        Args:
            recommended_items (list): List of recommended item IDs.
            relevant_items (set): Set of relevant item IDs.
            k (int): Number of top recommendations to consider.
            
        Returns:
            float: The Precision@K score.
        """
        if not relevant_items:
            return 0.0
        recommended_at_k = recommended_items[:k]
        hits = sum(1 for item in recommended_at_k if item in relevant_items)
        return hits / k

    def ndcg_at_k(self, recommended_items, relevant_items, k=5):
        """
        Compute Normalized Discounted Cumulative Gain (NDCG) at K.
        
        Args:
            recommended_items (list): List of recommended item IDs.
            relevant_items (set): Set of relevant item IDs.
            k (int): Number of top recommendations to consider.
            
        Returns:
            float: The NDCG@K score.
        """
        def dcg(scores):
            return sum((score / np.log2(idx + 2)) for idx, score in enumerate(scores))
        
        recommended_at_k = recommended_items[:k]
        gains = [1 if item in relevant_items else 0 for item in recommended_at_k]
        
        ideal_gains = sorted([1] * len(relevant_items) + [0] * (k - len(relevant_items)), reverse=True)
        
        actual_dcg = dcg(gains)
        ideal_dcg = dcg(ideal_gains[:k])
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def compute_user_metrics(self, test_data: pl.DataFrame, user_id: int, k=5, allow_read=False):
        '''
        Compute Precision@K and NDCG@K for a single user.

        Args:
            user_id (int): The user ID.
            k (int): Number of top recommendations to consider.

        Returns:
            tuple or None: (precision, ndcg) scores, or None if the user has no test interactions.
        '''
        relevant_items = set(
            test_data.filter(
                pl.col("user_id") == user_id)["article_id"].to_numpy())
        print(relevant_items)
        if not relevant_items:
            return None

        recommended_items = self.recommend_n_articles(user_id, n=k, allow_read=allow_read)
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        ndcg = self.ndcg_at_k(recommended_items, relevant_items, k)

        return precision, ndcg

    def evaluate_recommender(self, test_data: pl.DataFrame, k=5, n_jobs=-1, user_sample=None, allow_read=False):
        '''
        Evaluate the recommender using MAP@K and NDCG@K in parallel on a sample of users.

        Args:
            k (int): Number of top recommendations to consider.
            n_jobs (int): Number of parallel jobs for joblib.Parallel.
            user_sample (int or None): Number of users to sample for evaluation. If None, use all users.

        Returns:
            dict: A dictionary with MAP@K and NDCG@K scores.
        '''
        user_ids = self.impressions["user_id"].unique().to_numpy()

        if user_sample is not None and user_sample < len(user_ids):
            user_ids = np.random.choice(user_ids,
                                        size=user_sample,
                                        replace=False)

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.compute_user_metrics)(test_data, user_id, k, allow_read)
            for user_id in user_ids)
        results = [res for res in results if res is not None]

        if not results:
            return {"MAP@K": 0.0, "NDCG@K": 0.0}

        map_scores, ndcg_scores = zip(*results)

        return {
            "MAP@K": np.mean(map_scores),
            "NDCG@K": np.mean(ndcg_scores),
        }
