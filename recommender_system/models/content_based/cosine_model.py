import polars as pl
import numpy as np
from joblib import Parallel, delayed

class CosineModel:
    def __init__(self, train_behavior: pl.DataFrame, articles: pl.DataFrame, embeddings: pl.DataFrame):
        self.train_behavior = train_behavior
        self.articles_info = self.get_articles_info(articles, embeddings)
        # Build mapping from article_id to index for later use.
        self.article_id_to_index = {
            article_id: index
            for index, article_id in enumerate(self.articles_info["article_id"])
        }
        self.index_to_article_id = {
            index: article_id
            for index, article_id in enumerate(self.articles_info["article_id"])
        }
        # Precompute normalized embeddings for all articles.
        self.all_embeddings = self.get_embeddings_matrix(self.articles_info)
        self.all_embeddings_norm = self.normalize_matrix(self.all_embeddings)
        # Placeholder for precomputed user scores.
        self.scored_df = None

    def get_articles_info(self, articles: pl.DataFrame, embeddings: pl.DataFrame) -> pl.DataFrame:
        """
        Join article info with embeddings and cast key columns to float.
        """
        articles_info = articles.select([
            "article_id", "last_modified_time", "premium",
            "published_time", "category", "sentiment_score"
        ]).join(embeddings, on="article_id", how="inner")
        articles_info = articles_info.with_columns([
            pl.col("sentiment_score").cast(pl.Float32),
            pl.col("published_time").cast(pl.Float32),
            pl.col("last_modified_time").cast(pl.Float32),
            pl.col("premium").cast(pl.Float32),
            pl.col("category").cast(pl.Float32)
        ])
        return articles_info

    def get_embeddings_matrix(self, articles_info: pl.DataFrame) -> np.ndarray:
        """
        Extract the embedding matrix from articles_info.
        If there is a single embedding column (stored as a list), extract and stack it.
        Otherwise, assume embeddings are in multiple columns.
        """
        non_embedding_cols = [
            "article_id", "last_modified_time", "premium",
            "published_time", "category", "sentiment_score"
        ]
        embedding_cols = [col for col in articles_info.columns if col not in non_embedding_cols]
        
        if len(embedding_cols) == 1:
            # Assume embeddings are stored as a list column.
            embeddings_series = articles_info[embedding_cols[0]]
            embeddings_list = embeddings_series.to_list()
            embeddings_matrix = np.vstack(embeddings_list).astype(np.float32)
        else:
            # Embeddings are spread over multiple columns.
            embeddings_df = articles_info.select(embedding_cols)
            embeddings_matrix = embeddings_df.to_numpy().astype(np.float32)
            
        return embeddings_matrix

    def normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize each row of the given matrix using the L2 norm.
        """
        matrix = matrix.astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero.
        return matrix / norms

    def _get_user_profile(self, user_id: int) -> np.ndarray:
        """
        Compute a user profile vector as the average of normalized embeddings for articles the user interacted with.
        """
        user_articles = self.train_behavior.filter(pl.col("user_id") == user_id)
        # Only sort if the column exists.
        if "last_modified_time" in user_articles.columns:
            user_articles = user_articles.sort("last_modified_time", descending=True)
        user_articles = user_articles.join(self.articles_info, on="article_id", how="inner")
        
        if user_articles.height == 0:
            # If the user has no interactions, return a zero vector.
            return np.zeros(self.all_embeddings.shape[1])
        
        non_embedding_cols = [
            "article_id", "last_modified_time", "premium",
            "published_time", "category", "sentiment_score"
        ]
        embedding_cols = [col for col in self.articles_info.columns if col not in non_embedding_cols]
        
        if len(embedding_cols) == 1:
            # Handle list-type embedding.
            user_embeddings_list = user_articles[embedding_cols[0]].to_list()
            user_embeddings = np.vstack(user_embeddings_list).astype(np.float32)
        else:
            user_embeddings = user_articles.select(embedding_cols).to_numpy().astype(np.float32)
        
        user_embeddings_norm = self.normalize_matrix(user_embeddings)
        return np.mean(user_embeddings_norm, axis=0)

    def _compute_user_scores(self, user_id: int) -> pl.DataFrame:
        """
        Compute cosine similarity scores between the user's profile vector and all article embeddings.
        """
        user_profile = self._get_user_profile(user_id)
        norm = np.linalg.norm(user_profile)
        user_profile_norm = user_profile if norm == 0 else user_profile / norm
        scores = np.dot(self.all_embeddings_norm, user_profile_norm)
        rec_df = pl.DataFrame({
            "article_id": self.articles_info["article_id"],
            "score": scores
        })
        return rec_df.sort("score", descending=True)

    def fit(self):
        """
        Precompute and store the recommendation scores for every user in the training set.
        The result is stored in self.scored_df as a dictionary mapping user_id to their scored DataFrame.
        """
        self.scored_df = {}
        # Get unique user IDs from train_behavior.
        user_ids = self.train_behavior["user_id"].unique().to_list()
        for uid in user_ids:
            self.scored_df[uid] = self._compute_user_scores(uid)
    
    def get_scores(self, user_id: int, top_k: int) -> pl.DataFrame:
        """
        Returns the top_k similarity scores for the given user.
        If self.scored_df is available, it uses the precomputed scores.
        """
        if self.scored_df is not None and user_id in self.scored_df:
            return self.scored_df[user_id].head(top_k)
        else:
            return self._compute_user_scores(user_id).head(top_k)

    def recommend(self, user_id: int, n: int) -> list:
        """
        Returns a list of top_k recommended article IDs for the given user.
        """
        scores_df = self.get_scores(user_id, n)
        return scores_df["article_id"].to_list()
    


    def evaluate_recommender(self, test_behavior: pl.DataFrame, k: int = 5) -> dict:
        """
        Evaluate the recommender using MAP@K and NDCG@K on the provided test_behavior dataframe.

        Parameters
        ----------
        test_behavior : pl.DataFrame
            DataFrame containing test interactions with at least columns "user_id" and "article_id".
        k : int, optional
            Number of top recommendations to consider (default is 5).

        Returns
        -------
        dict
            Dictionary with keys "MAP@K" and "NDCG@K" corresponding to the average scores.
        """
        # Group test_behavior by user_id once
        grouped = test_behavior.group_by("user_id").agg(pl.col("article_id").list())
        test_dict = {row["user_id"]: set(row["article_id"]) for row in grouped.to_dicts()}

        def evaluate_user(uid, relevant_items):
            if not relevant_items:
                return None
            recommended = self.recommend(uid, k)
            # Define helper functions for metrics
            def average_precision_at_k(recommended, relevant, k):
                if not relevant:
                    return 0.0
                ap = 0.0
                num_hits = 0.0
                for i, item in enumerate(recommended[:k]):
                    if item in relevant:
                        num_hits += 1.0
                        ap += num_hits / (i + 1)
                return ap / min(len(relevant), k)
            
            def ndcg_at_k(recommended, relevant, k):
                dcg = 0.0
                for i, item in enumerate(recommended[:k]):
                    if item in relevant:
                        dcg += 1.0 / np.log2(i + 2)
                ideal_relevant_count = min(len(relevant), k)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_relevant_count))
                return dcg / idcg if idcg > 0 else 0.0

            ap = average_precision_at_k(recommended, relevant_items, k)
            ndcg = ndcg_at_k(recommended, relevant_items, k)
            return ap, ndcg

        # Get the list of user IDs present in the test set
        user_ids = list(test_dict.keys())
        
        # Use parallel processing to evaluate each user.
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_user)(uid, test_dict[uid]) for uid in user_ids
        )
        
        # Filter out users that returned None (if any)
        results = [res for res in results if res is not None]
        
        if not results:
            return {"MAP@K": 0.0, "NDCG@K": 0.0}
        
        map_scores, ndcg_scores = zip(*results)
        
        return {
            "MAP@K": np.mean(map_scores),
            "NDCG@K": np.mean(ndcg_scores)
        }

