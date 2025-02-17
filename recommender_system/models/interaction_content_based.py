import polars as pl
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

class SGDContentBased:
    def __init__(self, binary_interaction: pl.DataFrame, articles_embedding: pl.DataFrame, test_data: pl.DataFrame = None, batch_size: int = 1_000_000, n_components: int = 50):
        # Convert binary_interaction to lazy frame for potential optimizations.
        self.binary_interaction = binary_interaction.lazy()
        # Convert articles_embedding to an eager DataFrame if it's lazy.
        if hasattr(articles_embedding, "collect"):
            self.articles_embedding = articles_embedding.collect()
        else:
            self.articles_embedding = articles_embedding

        self.test_data = test_data  # Test data for evaluation (Polars DataFrame)
        self.batch_size = batch_size
        self.model = SGDClassifier(loss="log_loss", max_iter=1000, learning_rate="optimal")
        self.first_batch = True
        self.pca = PCA(n_components=n_components)

    def _prepare_features(self, df: pl.DataFrame) -> np.ndarray:
        """
        Convert a Polars DataFrame to a NumPy array of features.
        If there is a single column that contains list-like elements,
        it is assumed to be an embedding column and is converted using np.vstack.
        """
        # Drop non-feature columns and convert to pandas DataFrame.
        X = df.drop(["user_id", "article_id", "clicked"]).to_pandas()
        # If there's only one column, it might be a list/array column.
        if X.shape[1] == 1:
            X = np.vstack(X.iloc[:, 0])
        else:
            X = X.values
        return X

    def fit(self):
        """
        Train the model in mini-batches to avoid memory overload.
        """
        first_pca_fit = True
        
        # Collect binary interactions in streaming mode.
        binary_interaction_df = self.binary_interaction.collect(streaming=True)
        total_rows = binary_interaction_df.height

        for start in range(0, total_rows, self.batch_size):
            # Slice a batch from the full DataFrame.
            batch = binary_interaction_df.slice(start, self.batch_size)
            # Join with the pre-collected article embeddings.
            batch_embeddings = batch.join(self.articles_embedding, on="article_id", how="inner")
            if batch_embeddings.is_empty():
                continue  # Skip if no data in the batch.

            # Prepare feature array and labels.
            X_batch = self._prepare_features(batch_embeddings)
            y_batch = batch_embeddings["clicked"].to_pandas()

            # Apply PCA for feature reduction.
            if first_pca_fit:
                X_batch = self.pca.fit_transform(X_batch)
                first_pca_fit = False
            else:
                X_batch = self.pca.transform(X_batch)

            # Train using online learning via partial_fit.
            if self.first_batch:
                self.model.partial_fit(X_batch, y_batch, classes=[0, 1])
                self.first_batch = False
            else:
                self.model.partial_fit(X_batch, y_batch)

        print("Training complete!")

    def recommend(self, user_id: int, n_recommendations: int):
        """
        Generate recommendations for a given user based on predicted probability of click.
        """
        # Filter interactions for the given user that haven't been clicked.
        user_articles = self.binary_interaction.filter(pl.col("user_id") == user_id) \
                                                 .filter(pl.col("clicked") == 0) \
                                                 .collect()
        # Join with article embeddings.
        user_articles = user_articles.join(self.articles_embedding, on="article_id", how="inner")
        
        if user_articles.is_empty():
            return pl.DataFrame()  # Return empty if no recommendations available.
        
        # Prepare feature array.
        X_user = self._prepare_features(user_articles)
        X_user = self.pca.transform(X_user)
        
        # Predict scores using the trained model.
        predictions = self.model.predict_proba(X_user)[:, 1]  # Probability of click.
        
        # Attach predictions and sort recommendations.
        user_articles = user_articles.with_columns(pl.Series("prediction", predictions))
        user_articles = user_articles.sort("prediction", descending=True)

        return user_articles.head(n_recommendations)

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
            return sum(score / np.log2(idx + 2) for idx, score in enumerate(scores))
        
        recommended_at_k = recommended_items[:k]
        gains = [1 if item in relevant_items else 0 for item in recommended_at_k]
        
        # Ideal gains: assume the best possible ordering (all relevant items at the top).
        ideal_gains = sorted([1] * min(len(relevant_items), k) + [0] * (k - min(len(relevant_items), k)), reverse=True)
        
        actual_dcg = dcg(gains)
        ideal_dcg = dcg(ideal_gains)
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def compute_user_metrics(self, user_id, k=5):
        """
        Compute Precision@K and NDCG@K for a single user.
        
        Args:
            user_id (int): The user ID.
            k (int): Number of top recommendations to consider.
            
        Returns:
            tuple or None: (precision, ndcg) scores, or None if the user has no test interactions.
        """
        if self.test_data is None:
            raise ValueError("Test data is not provided for evaluation.")

        # Filter test data for the given user.
        user_test = self.test_data.filter(pl.col("user_id") == user_id)
        if user_test.is_empty():
            return None

        relevant_items = set(user_test["article_id"].to_list())
        if not relevant_items:
            return None

        # Get recommendations and extract recommended article IDs.
        recommended_df = self.recommend(user_id, n_recommendations=k)
        if recommended_df.is_empty():
            return None
        recommended_items = recommended_df["article_id"].to_list()

        precision = self.precision_at_k(recommended_items, relevant_items, k)
        ndcg = self.ndcg_at_k(recommended_items, relevant_items, k)

        return precision, ndcg

    def evaluate_recommender(self, k=5, n_jobs=-1, user_sample=None):
        """
        Evaluate the recommender using MAP@K and NDCG@K in parallel on a sample of users.
        
        Args:
            k (int): Number of top recommendations to consider.
            n_jobs (int): Number of parallel jobs for joblib.Parallel.
            user_sample (int or None): Number of users to sample for evaluation. If None, use all users.
            
        Returns:
            dict: A dictionary with MAP@K and NDCG@K scores.
        """
        if self.test_data is None:
            raise ValueError("Test data is not provided for evaluation.")

        user_ids = self.test_data["user_id"].unique().to_list()

        if user_sample is not None and user_sample < len(user_ids):
            user_ids = np.random.choice(user_ids, size=user_sample, replace=False)
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.compute_user_metrics)(user_id, k) for user_id in user_ids
        )
        # Filter out users with no test interactions.
        results = [res for res in results if res is not None]
        
        if not results:
            return {"MAP@K": 0.0, "NDCG@K": 0.0}
        
        map_scores, ndcg_scores = zip(*results)
        
        return {
            "MAP@K": np.mean(map_scores),
            "NDCG@K": np.mean(ndcg_scores),
        }
