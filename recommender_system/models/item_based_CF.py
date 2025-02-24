import polars as pl
import numpy as np
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed


class ItemItemCollaborativeRecommender:
    """
    Implements an item–item collaborative recommender using user interaction data.

    This model builds an item similarity matrix based on cosine similarity computed
    from interaction scores. It then uses this matrix to generate recommendations for users.
    """

    def __init__(self, interactions: pl.DataFrame, binary_model: bool = False):
        """
        Initialize the Item-Item Collaborative Recommender.

        Parameters
        ----------
        interactions : pl.DataFrame
            DataFrame containing user interactions with articles.
        binary_model : bool, optional
            Flag indicating whether to use only binary interactions (True) or detailed
            interaction scores (False). Default is False.
        """
        self.interactions = interactions
        self.binary_model = binary_model
        self.item_similarity_matrix = {}

    def add_interaction_scores(self, scroll_weight: float = 1.0, readtime_weight: float = 1.0) -> pl.DataFrame:
        """
        Compute and add an `interaction_score` column to the interactions DataFrame.

        The interaction score is computed as a weighted sum of the `max_scroll` and 
        `total_readtime` columns.

        Parameters
        ----------
        scroll_weight : float, optional
            Weight applied to the `max_scroll` column (default is 1.0).
        readtime_weight : float, optional
            Weight applied to the `total_readtime` column (default is 1.0).

        Returns
        -------
        pl.DataFrame
            The updated DataFrame with an added `interaction_score` column.
        """
        # Compute the interaction score and add it as a new column.
        self.interactions = self.interactions.with_columns(
            (
                pl.col("max_scroll") * scroll_weight +
                pl.col("total_readtime") * readtime_weight
            ).alias("interaction_score")
        )
        return self.interactions

    def build_item_similarity_matrix(self, sim_size: int = 10):
        """
        Build an item similarity matrix using cosine similarity based on interaction scores.

        This method creates a pivot table from the interactions (using binary values if the
        binary model is enabled), computes the cosine similarity between item vectors, and
        retains the top similar items for each article.

        Parameters
        ----------
        sim_size : int, optional
            Number of similar items to retain per item (default is 10).

        Returns
        -------
        dict
            Dictionary mapping each article ID to a list of tuples (similar_article_id, similarity_score).
        """
        # Create a pivot table where values are either binary (1) or the computed interaction score.
        item_user_matrix = self.interactions.with_columns(
            pl.lit(1).alias("interaction_score") if self.binary_model else pl.col("interaction_score")
        ).pivot(values="interaction_score", index="article_id", columns="user_id").fill_null(0)

        # Extract the list of item IDs and corresponding vectors.
        item_ids = item_user_matrix["article_id"].to_list()
        item_vectors = item_user_matrix.drop("article_id").to_numpy()

        # Compute the cosine similarity matrix.
        similarity_matrix = 1 - squareform(pdist(item_vectors, metric='cosine'))
        # Retrieve indices of the top similar items for each item, excluding itself.
        top_similarities = np.argsort(-similarity_matrix, axis=1)[:, 1:sim_size + 1]

        # Build the item similarity dictionary.
        self.item_similarity_matrix = {
            item_ids[i]: [(item_ids[j], similarity_matrix[i, j]) for j in top_similarities[i]]
            for i in range(len(item_ids))
        }
        return self.item_similarity_matrix

    def fit(self):
        """
        Fit the item–item collaborative recommender by building the item similarity matrix.

        Returns
        -------
        dict
            The item similarity matrix.
        """
        # Compute interaction scores if using detailed interactions.
        if not self.binary_model:
            self.add_interaction_scores()
        return self.build_item_similarity_matrix()

    def recommend_n_articles(self, user_id: int, n: int, allow_read_articles: bool = False) -> list[int]:
        """
        Recommend the top n articles for a user based on similar items.

        The method aggregates similarity scores from the articles the user has interacted with
        and returns the articles with the highest cumulative scores, optionally excluding
        articles the user has already read.

        Parameters
        ----------
        user_id : int
            The identifier of the user.
        n : int
            The number of articles to recommend.
        allow_read_articles : bool, optional
            Flag indicating whether to include articles already read by the user (default is False).

        Returns
        -------
        list of int
            List of recommended article IDs.
        """
        # Retrieve articles that the user has interacted with.
        user_articles = self.interactions.filter(pl.col("user_id") == user_id)["article_id"].to_list()
        article_scores = {}

        # Sum similarity scores from each article the user has read.
        for article in user_articles:
            if article in self.item_similarity_matrix:
                for similar_article, similarity in self.item_similarity_matrix[article]:
                    if not allow_read_articles and similar_article in user_articles:
                        continue  # Skip articles already read.
                    article_scores[similar_article] = article_scores.get(similar_article, 0) + similarity

        # Return the top n articles sorted by cumulative similarity.
        return [article for article, _ in sorted(article_scores.items(), key=lambda x: x[1], reverse=True)[:n]]

    def precision_at_k(self, recommended_items, relevant_items, k: int = 5) -> float:
        """
        Compute the Precision@K of the recommendations.

        Parameters
        ----------
        recommended_items : list
            List of recommended article IDs.
        relevant_items : set
            Set of relevant article IDs.
        k : int, optional
            Number of top recommendations to consider (default is 5).

        Returns
        -------
        float
            The Precision@K score.
        """
        if not relevant_items:
            return 0.0
        recommended_at_k = recommended_items[:k]
        hits = sum(1 for item in recommended_at_k if item in relevant_items)
        return hits / k

    def ndcg_at_k(self, recommended_items, relevant_items, k: int = 5) -> float:
        """
        Compute Normalized Discounted Cumulative Gain (NDCG) at K.

        Parameters
        ----------
        recommended_items : list
            List of recommended article IDs.
        relevant_items : set
            Set of relevant article IDs.
        k : int, optional
            Number of top recommendations to consider (default is 5).

        Returns
        -------
        float
            The NDCG@K score.
        """
        def dcg(scores):
            return sum(score / np.log2(idx + 2) for idx, score in enumerate(scores))
        
        recommended_at_k = recommended_items[:k]
        gains = [1 if item in relevant_items else 0 for item in recommended_at_k]
        # Compute ideal gains assuming the best ordering.
        ideal_gains = sorted([1] * len(relevant_items) + [0] * (k - len(relevant_items)), reverse=True)
        actual_dcg = dcg(gains)
        ideal_dcg = dcg(ideal_gains[:k])
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def compute_user_metrics(self, test_data: pl.DataFrame, user_id: int, k: int = 5,
                             allow_read_articles: bool = False):
        """
        Compute Precision@K and NDCG@K for a single user.

        Parameters
        ----------
        test_data : pl.DataFrame
            Test data containing user interactions.
        user_id : int
            The identifier of the user.
        k : int, optional
            Number of top recommendations to consider (default is 5).
        allow_read_articles : bool, optional
            Flag indicating whether to include articles already read by the user (default is False).

        Returns
        -------
        tuple or None
            A tuple (precision, ndcg) if test interactions exist; otherwise, None.
        """
        # Filter test data to retrieve the user's relevant articles.
        relevant_items = set(
            test_data.filter(pl.col("user_id") == user_id)["article_id"].to_numpy()
        )
        print(relevant_items)
        if not relevant_items:
            return None

        # Generate recommendations for the user.
        recommended_items = self.recommend_n_articles(user_id, n=k, allow_read_articles=allow_read_articles)
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        ndcg = self.ndcg_at_k(recommended_items, relevant_items, k)
        return precision, ndcg

    def evaluate_recommender(self, test_data: pl.DataFrame, k: int = 5, n_jobs: int = -1,
                             user_sample: int = None, allow_read_articles: bool = False) -> dict:
        """
        Evaluate the recommender using MAP@K and NDCG@K over a sample of users.

        Parameters
        ----------
        test_data : pl.DataFrame
            Test data containing user interactions.
        k : int, optional
            Number of top recommendations to consider (default is 5).
        n_jobs : int, optional
            Number of parallel jobs for evaluation (default is -1, which uses all available processors).
        user_sample : int or None, optional
            Number of users to sample for evaluation. If None, evaluates all users.
        allow_read_articles : bool, optional
            Flag indicating whether to include articles already read by the user (default is False).

        Returns
        -------
        dict
            Dictionary with keys "MAP@K" and "NDCG@K" representing the average scores.
        """
        user_ids = self.interactions["user_id"].unique().to_numpy()

        if user_sample is not None and user_sample < len(user_ids):
            user_ids = np.random.choice(user_ids, size=user_sample, replace=False)

        # Evaluate metrics in parallel across users.
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.compute_user_metrics)(test_data, user_id, k, allow_read_articles)
            for user_id in user_ids
        )
        results = [res for res in results if res is not None]

        if not results:
            return {"MAP@K": 0.0, "NDCG@K": 0.0}

        map_scores, ndcg_scores = zip(*results)
        return {
            "MAP@K": np.mean(map_scores),
            "NDCG@K": np.mean(ndcg_scores),
        }
