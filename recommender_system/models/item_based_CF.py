import polars as pl
import numpy as np
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed

class ItemItemCollaborativeRecommender:
    def __init__(self, interactions: pl.DataFrame, binary_model=False):
        '''
        Initialize the Item-Item Collaborative Recommender with a user-item dataframe.
        
        Parameters
        ----------
        interactions : pl.DataFrame
            A DataFrame containing user interactions with articles.
        binary_model : bool
            Whether or not we use only article interactions (True) or interaction details as well (False).
        '''
        self.interactions = interactions
        self.binary_model = binary_model
        self.item_similarity_matrix = {}

    def add_interaction_scores(self, scroll_weight: float = 1.0, readtime_weight: float = 1.0) -> pl.DataFrame:
        """
        Computes and adds an `interaction_score` column to the `interactions` DataFrame.
        """
        self.interactions = self.interactions.with_columns(
            (
                pl.col("max_scroll") * scroll_weight +
                pl.col("total_readtime") * readtime_weight
            ).alias("interaction_score")
        )
        return self.interactions

    def build_item_similarity_matrix(self, sim_size=10):
        '''
        Builds an item similarity matrix using cosine similarity based on interaction scores.
        Each item contains the `sim_size` most similar items, sorted by similarity.
        '''
        item_user_matrix = self.interactions.with_columns(
            pl.lit(1).alias("interaction_score") if self.binary_model else pl.col("interaction_score")
        ).pivot(values="interaction_score", index="article_id", columns="user_id").fill_null(0)

        item_ids = item_user_matrix["article_id"].to_list()
        item_vectors = item_user_matrix.drop("article_id").to_numpy()

        similarity_matrix = 1 - squareform(pdist(item_vectors, metric='cosine'))
        top_similarities = np.argsort(-similarity_matrix, axis=1)[:, 1:sim_size + 1]

        self.item_similarity_matrix = {
            item_ids[i]: [(item_ids[j], similarity_matrix[i, j]) for j in top_similarities[i]]
            for i in range(len(item_ids))
        }
        return self.item_similarity_matrix

    def fit(self):
        '''
        Fits the Item-Item Collaborative Recommender model by building the item similarity matrix.
        '''
        self.add_interaction_scores() if not self.binary_model else None
        return self.build_item_similarity_matrix()

    def recommend_n_articles(self, user_id: int, n: int, allow_read_articles=False) -> list[int]:
        '''
        Recommend the top n articles for a user based on similar items.
        '''
        user_articles = self.interactions.filter(pl.col("user_id") == user_id)["article_id"].to_list()
        article_scores = {}

        for article in user_articles:
            if article in self.item_similarity_matrix:
                for similar_article, similarity in self.item_similarity_matrix[article]:
                    if not allow_read_articles and similar_article in user_articles:
                        continue
                    article_scores[similar_article] = article_scores.get(similar_article, 0) + similarity

        return [article for article, _ in sorted(article_scores.items(), key=lambda x: x[1], reverse=True)[:n]]

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
