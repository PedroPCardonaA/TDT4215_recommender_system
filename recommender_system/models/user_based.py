import polars as pl
import numpy as np
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed

class CollaborativeRecommender:
    def __init__(self, interactions: pl.DataFrame, binary_model = False):
        '''
        Initialize the CollaborativeRecommender with a user-item dataframe.

        Parameters
        ----------
        interactions : pl.DataFrame
            A DataFrame containing user interactions with articles.
        binary_model : bool
            Whether or not we use only reading articles for user similarity (True) or user interaction details as well (False)
        '''
        self.interactions = interactions
        self.binary_model = binary_model
        self.user_similarity_matrix = {}

    def add_interaction_scores(self, scroll_weight: float = 1.0, readtime_weight: float = 1.0) -> pl.DataFrame:
        """
        Computes and adds an `interaction_score` column to the `interactions` DataFrame.

        The interaction score is calculated as a weighted sum of the `max_scroll` and `total_readtime` columns:
        
            interaction_score = (max_scroll * scroll_weight) + (total_readtime * readtime_weight)

        Parameters
        ----------
        scroll_weight : float, optional
            The weight assigned to the `max_scroll` column (default is 1.0).
        readtime_weight : float, optional
            The weight assigned to the `total_readtime` column (default is 1.0).

        Returns
        -------
        pl.DataFrame
            A DataFrame with an additional `interaction_score` column.
        """
        self.interactions = self.interactions.with_columns(
            (
                pl.col("max_scroll") * scroll_weight +
                pl.col("total_readtime") * readtime_weight
            ).alias("interaction_score")
        )
        return self.interactions


    def build_user_similarity_matrix(self, sim_size=10):
        '''
        Builds a user similarity matrix using cosine similarity based on interaction scores.
        Each user contains the `sim_size` most similar users, sorted by similarity.

        Parameters
        ----------
        sim_size : int
            How many other similar users should be saved for every user in the matrix

        Returns
        -------
        dict
            A dictionary of lists where the keys are user IDs and the values in the lists are 
            `sim_size` instances of the most similar users, sorted by similarity.
        '''
        # Create user-item matrix
        user_item_matrix = self.interactions.with_columns(
            pl.lit(1).alias("interaction_score") if self.binary_model else pl.col("interaction_score")
        ).pivot(values="interaction_score", index="user_id", columns="article_id").fill_null(0)

        user_ids = user_item_matrix["user_id"].to_list()
        user_vectors = user_item_matrix.drop("user_id").to_numpy()

        # Compute similarity matrix and get top `sim_size` similar users
        similarity_matrix = 1 - squareform(pdist(user_vectors, metric='cosine'))
        top_similarities = np.argsort(-similarity_matrix, axis=1)[:, 1:sim_size + 1]

        # Store the most similar users for each user
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
        self.add_interaction_scores() if not self.binary_model else None

        return self.build_user_similarity_matrix()

    def recommend_n_articles(self, user_id: int, n: int, allow_read_articles=False) -> list[int]:
        '''
        Recommend the top n articles for a user based on similar users' activity, excluding already read articles unless allowed.

        Parameters
        ----------
        user_id : int
            The ID of the user for whom to make predictions.
        n : int
            The number of articles to recommend.
        allow_read_articles : bool
            Whether already read articles can be recommended.

        Returns
        -------
        list[int]
            A list of article IDs predicted to be most liked by the user.
        '''
        if user_id not in self.user_similarity_matrix:
            return []  # Return empty list if user not found

        # Get the articles the user has already read
        read_articles = set(
            self.interactions.filter(pl.col("user_id") == user_id)["article_id"].to_list()
        )

        # Get similar users' article interactions
        similar_users = [uid for uid, _ in self.user_similarity_matrix[user_id]]
        similar_user_articles = self.interactions.filter(pl.col("user_id").is_in(similar_users))

        # Aggregate interaction scores for each article
        article_scores = similar_user_articles.groupby("article_id").agg(
            pl.len().alias("total_score") if self.binary_model else pl.col("interaction_score").sum().alias("total_score")
        )

        # Filter out articles the user has already read (unless allowed)
        if not allow_read_articles:
            article_scores = article_scores.filter(~pl.col("article_id").is_in(read_articles))

        # Sort by total score and select top n articles
        top_articles = article_scores.sort("total_score", descending=True).head(n)

        return top_articles["article_id"].to_list()

    def precision_at_k(self, recommended_items, relevant_items, k=5):
        '''
        Compute the Precision@K of our model.
        
        Parameters
        ----------
        recommended_items : list
            List of recommended item IDs.
        relevant_items : set
            Set of relevant item IDs.
        k : int 
            Number of top recommendations to consider.

        Returns
        -------
        float
            The Precision@K score.
        '''
        if not relevant_items:
            return 0.0
        recommended_at_k = recommended_items[:k]
        hits = sum(1 for item in recommended_at_k if item in relevant_items)
        return hits / k

    def ndcg_at_k(self, recommended_items, relevant_items, k=5):
        '''
        Compute Normalized Discounted Cumulative Gain (NDCG) at K.
        
        Parameters
        ----------
        recommended_items : list
            List of recommended item IDs.
        relevant_items : set
            Set of relevant item IDs.
        k : int 
            Number of top recommendations to consider.

        Returns
        -------
        float
            The NDCG@K score.
        '''
        def dcg(scores):
            return sum((score / np.log2(idx + 2)) for idx, score in enumerate(scores))
        
        recommended_at_k = recommended_items[:k]
        gains = [1 if item in relevant_items else 0 for item in recommended_at_k]
        
        ideal_gains = sorted([1] * len(relevant_items) + [0] * (k - len(relevant_items)), reverse=True)
        
        actual_dcg = dcg(gains)
        ideal_dcg = dcg(ideal_gains[:k])
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def compute_user_metrics(self, test_data: pl.DataFrame, user_id: int, k=5, allow_read_articles=False):
        '''
        Compute Precision@K and NDCG@K for a single user.

        Parameters
        ----------
        user_id : int
            The user ID.
        k : int
            Number of top recommendations to consider.
        
        Returns
        -------
        tuple or None
            (precision, ndcg) scores, or None if the user has no test interactions.

        '''
        relevant_items = set(
            test_data.filter(
                pl.col("user_id") == user_id)["article_id"].to_numpy())
        print(relevant_items)
        if not relevant_items:
            return None

        recommended_items = self.recommend_n_articles(user_id, n=k, allow_read_articles=allow_read_articles)
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        ndcg = self.ndcg_at_k(recommended_items, relevant_items, k)

        return precision, ndcg

    def evaluate_recommender(self, test_data: pl.DataFrame, k=5, n_jobs=-1, user_sample=None, allow_read_articles=False):
        '''
        Evaluate the recommender using MAP@K and NDCG@K in parallel on a sample of users.

        
        Parameters
        ----------
        k : int
            Number of top recommendations to consider.
        n_jobs : int
            Number of parallel jobs for joblib.Parallel.
        user_sample : int or None
            Number of users to sample for evaluation. If None, use all users.
        
        Returns
        -------
        dict 
            A dictionary with MAP@K and NDCG@K scores.

        '''
        user_ids = self.interactions["user_id"].unique().to_numpy()

        if user_sample is not None and user_sample < len(user_ids):
            user_ids = np.random.choice(user_ids,
                                        size=user_sample,
                                        replace=False)

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.compute_user_metrics)(test_data, user_id, k, allow_read_articles)
            for user_id in user_ids)
        results = [res for res in results if res is not None]

        if not results:
            return {"MAP@K": 0.0, "NDCG@K": 0.0}

        map_scores, ndcg_scores = zip(*results)

        return {
            "MAP@K": np.mean(map_scores),
            "NDCG@K": np.mean(ndcg_scores),
        }
