import polars as pl
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from joblib import Parallel, delayed


class SGDContentBased:
    """
    Implements a content-based recommender using an SGD classifier with PCA-based feature reduction.

    The model is trained in mini-batches to avoid memory overload. It uses a binary interaction
    DataFrame for training, an articles embedding DataFrame for feature extraction, and an optional
    test DataFrame for evaluation.
    """

    def __init__(self, binary_interaction: pl.DataFrame, articles_embedding: pl.DataFrame,
                 test_data: pl.DataFrame = None, batch_size: int = 1_000_000, n_components: int = 50):
        """
        Initialize the recommender and set up model components.

        Parameters
        ----------
        binary_interaction : pl.DataFrame
            DataFrame of binary interactions. Converted to a lazy frame for optimization.
        articles_embedding : pl.DataFrame
            DataFrame containing article embeddings. If lazy, it is collected eagerly.
        test_data : pl.DataFrame, optional
            Test data for evaluation.
        batch_size : int, optional
            Number of rows per mini-batch for training (default is 1,000,000).
        n_components : int, optional
            Number of principal components for feature reduction via PCA (default is 50).

        Attributes
        ----------
        binary_interaction : LazyFrame
            The binary interaction data in lazy mode.
        articles_embedding : pl.DataFrame
            Eager DataFrame of article embeddings.
        test_data : pl.DataFrame or None
            Test data for evaluation.
        batch_size : int
            Size of each training mini-batch.
        model : SGDClassifier
            SGD classifier model configured for logistic loss.
        first_batch : bool
            Flag to indicate the first training batch for initial partial_fit.
        pca : PCA
            PCA transformer for reducing feature dimensionality.
        """
        # Convert binary_interaction to a lazy frame for optimization.
        self.binary_interaction = binary_interaction.lazy()
        # Convert articles_embedding to an eager DataFrame if it is lazy.
        if hasattr(articles_embedding, "collect"):
            self.articles_embedding = articles_embedding.collect()
        else:
            self.articles_embedding = articles_embedding

        self.test_data = test_data  # Store test data for evaluation.
        self.batch_size = batch_size
        self.model = SGDClassifier(loss="log_loss", max_iter=1000, learning_rate="optimal")
        self.first_batch = True
        self.pca = PCA(n_components=n_components)

    def _prepare_features(self, df: pl.DataFrame) -> np.ndarray:
        """
        Convert a Polars DataFrame to a NumPy array of features.

        If the DataFrame contains a single column with list-like elements (assumed to be an
        embedding column), the method utilizes np.vstack to convert it.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame from which to extract features. Expects columns other than "user_id",
            "article_id", and "clicked" to be feature data.

        Returns
        -------
        np.ndarray
            Array of features.
        """
        # Drop non-feature columns and convert the remainder to a pandas DataFrame.
        X = df.drop(["user_id", "article_id", "clicked"]).to_pandas()
        # Utilize np.vstack if a single column contains list-like elements.
        if X.shape[1] == 1:
            X = np.vstack(X.iloc[:, 0])
        else:
            X = X.values
        return X

    def fit(self):
        """
        Train the SGD classifier in mini-batches with PCA-based feature reduction.

        The method collects the binary interaction data in streaming mode, processes it in
        mini-batches, applies PCA for feature reduction, and trains the model using partial_fit.
        """
        first_pca_fit = True

        # Collect binary interactions in streaming mode.
        binary_interaction_df = self.binary_interaction.collect(streaming=True)
        total_rows = binary_interaction_df.height

        for start in range(0, total_rows, self.batch_size):
            # Slice a mini-batch from the full DataFrame.
            batch = binary_interaction_df.slice(start, self.batch_size)
            # Join the batch with the pre-collected article embeddings.
            batch_embeddings = batch.join(self.articles_embedding, on="article_id", how="inner")
            if batch_embeddings.is_empty():
                continue  # Skip the batch if it contains no data.

            # Prepare the feature array and corresponding labels.
            X_batch = self._prepare_features(batch_embeddings)
            y_batch = batch_embeddings["clicked"].to_pandas()

            # Apply PCA for feature reduction.
            if first_pca_fit:
                X_batch = self.pca.fit_transform(X_batch)
                first_pca_fit = False
            else:
                X_batch = self.pca.transform(X_batch)

            # Train the model using online learning via partial_fit.
            if self.first_batch:
                self.model.partial_fit(X_batch, y_batch, classes=[0, 1])
                self.first_batch = False
            else:
                self.model.partial_fit(X_batch, y_batch)

        print("Training complete!")

    def recommend(self, user_id: int, n_recommendations: int):
        """
        Generate recommendations for a given user based on the predicted probability of click.

        Parameters
        ----------
        user_id : int
            Identifier for the user.
        n_recommendations : int
            Number of recommendations to return.

        Returns
        -------
        pl.DataFrame
            DataFrame of recommended articles sorted by predicted click probability. If no
            recommendations are available, returns an empty DataFrame.
        """
        # Filter interactions for the given user where "clicked" equals 0.
        user_articles = self.binary_interaction.filter(pl.col("user_id") == user_id) \
                                               .filter(pl.col("clicked") == 0) \
                                               .collect()
        # Join the user articles with the article embeddings.
        user_articles = user_articles.join(self.articles_embedding, on="article_id", how="inner")

        if user_articles.is_empty():
            return pl.DataFrame()  # Return an empty DataFrame if no recommendations exist.

        # Prepare the feature array from the user's articles.
        X_user = self._prepare_features(user_articles)
        X_user = self.pca.transform(X_user)

        # Predict click probabilities using the trained model.
        predictions = self.model.predict_proba(X_user)[:, 1]  # Extract probability of click.

        # Append predictions as a new column and sort recommendations by predicted probability.
        user_articles = user_articles.with_columns(pl.Series("prediction", predictions))
        user_articles = user_articles.sort("prediction", descending=True)

        return user_articles.head(n_recommendations)

    def precision_at_k(self, recommended_items, relevant_items, k=5):
        """
        Compute Precision@K for a set of recommendations.

        Parameters
        ----------
        recommended_items : list
            List of recommended item IDs.
        relevant_items : set
            Set of relevant item IDs.
        k : int, optional
            Number of top recommendations to consider (default is 5).

        Returns
        -------
        float
            Precision@K score.
        """
        if not relevant_items:
            return 0.0
        recommended_at_k = recommended_items[:k]
        hits = sum(1 for item in recommended_at_k if item in relevant_items)
        return hits / k

    def ndcg_at_k(self, recommended_items, relevant_items, k=5):
        """
        Compute Normalized Discounted Cumulative Gain (NDCG) at K for a set of recommendations.

        Parameters
        ----------
        recommended_items : list
            List of recommended item IDs.
        relevant_items : set
            Set of relevant item IDs.
        k : int, optional
            Number of top recommendations to consider (default is 5).

        Returns
        -------
        float
            NDCG@K score.
        """
        def dcg(scores):
            return sum(score / np.log2(idx + 2) for idx, score in enumerate(scores))

        recommended_at_k = recommended_items[:k]
        gains = [1 if item in relevant_items else 0 for item in recommended_at_k]

        # Compute ideal gains assuming the best possible ordering.
        ideal_gains = sorted([1] * min(len(relevant_items), k) + [0] * (k - min(len(relevant_items), k)), reverse=True)

        actual_dcg = dcg(gains)
        ideal_dcg = dcg(ideal_gains)

        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def compute_user_metrics(self, user_id, k=5):
        """
        Compute Precision@K and NDCG@K for a single user using test data.

        Parameters
        ----------
        user_id : int
            Identifier for the user.
        k : int, optional
            Number of top recommendations to consider (default is 5).

        Returns
        -------
        tuple or None
            Tuple of (precision, ndcg) scores if test interactions exist; otherwise, None.

        Raises
        ------
        ValueError
            If test data is not provided.
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

        # Generate recommendations and extract the recommended article IDs.
        recommended_df = self.recommend(user_id, n_recommendations=k)
        if recommended_df.is_empty():
            return None
        recommended_items = recommended_df["article_id"].to_list()

        precision = self.precision_at_k(recommended_items, relevant_items, k)
        ndcg = self.ndcg_at_k(recommended_items, relevant_items, k)

        return precision, ndcg

    def evaluate_recommender(self, k=5, n_jobs=-1, user_sample=None, random_seed=42):
        """
        Evaluate the recommender across multiple users using MAP@K and NDCG@K.

        Parameters
        ----------
        k : int, optional
            Number of top recommendations to consider (default is 5).
        n_jobs : int, optional
            Number of parallel jobs for evaluation (default is -1 to use all processors).
        user_sample : int or None, optional
            Number of users to sample for evaluation. If None, evaluates all users in test data.

        Returns
        -------
        dict
            Dictionary with keys "MAP@K" and "NDCG@K" representing the average scores.

        Raises
        ------
        ValueError
            If test data is not provided for evaluation.
        """
        np.random.seed(random_seed)
        if self.test_data is None:
            raise ValueError("Test data is not provided for evaluation.")

        user_ids = self.test_data["user_id"].unique().to_list()

        if user_sample is not None and user_sample < len(user_ids):
            user_ids = np.random.choice(user_ids, size=user_sample, replace=False)

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.compute_user_metrics)(user_id, k) for user_id in user_ids
        )
        # Filter out users without test interactions.
        results = [res for res in results if res is not None]

        if not results:
            return {"MAP@K": 0.0, "NDCG@K": 0.0}

        map_scores, ndcg_scores = zip(*results)

        return {
            "MAP@K": np.mean(map_scores),
            "NDCG@K": np.mean(ndcg_scores),
        }
    
    def aggregate_diversity(self, item_df, k=5, user_sample=None, random_seed=42):

        np.random.seed(random_seed)

        users = np.array(self.user_ids)

        if user_sample is not None and user_sample < len(users):
            users = np.random.choice(users, size=user_sample, replace=False)

        recommended_items = set()
        for user_id in users:
            recommended_items.update(self.recommend(user_id, n=k))

        total_items = set(item_df["article_id"].to_numpy())
        aggregate_diversity = len(recommended_items) / len(total_items) if total_items else 0.0

        return aggregate_diversity
    
    def gini_coefficient(self, k=5, user_sample=None, random_seed=42):
        """
        Compute the Gini coefficient to measure the concentration of recommendations.

        A Gini coefficient of 0 means that recommendations are equally distributed across items,
        whereas a Gini coefficient closer to 1 means that recommendations are highly concentrated
        on a small number of items (i.e., strong popularity bias).

        This version considers the full catalog of articles from self.articles_embedding,
        assigning a count of 0 to items that are never recommended.

        Parameters
        ----------
        k : int, optional
            Number of top recommendations per user (default is 5).
        user_sample : int, optional
            Number of users to sample for evaluation (if None, all users are evaluated).
        random_seed : int, optional
            Seed for reproducibility when sampling users (default is 42).

        Returns
        -------
        float
            The Gini coefficient of the item recommendation distribution.
        """
        np.random.seed(random_seed)
        # Get unique user IDs from the binary interaction data.
        users = np.array(self.binary_interaction["user_id"].unique().to_list())
        if user_sample is not None and user_sample < len(users):
            users = np.random.choice(users, size=user_sample, replace=False)

        recommended_items = []
        for user_id in users:
            rec_df = self.recommend(user_id, n_recommendations=k)
            if not rec_df.is_empty():
                recommended_items.extend(rec_df["article_id"].to_list())

        if not recommended_items:
            return 0.0

        # Count recommendations for items that were recommended.
        rec_counts = pl.DataFrame({"article_id": recommended_items}).group_by("article_id") \
            .agg(pl.len().alias("count"))
        
        # Create a DataFrame for the full catalog using article IDs from articles_embedding.
        full_catalog = pl.DataFrame({"article_id": self.articles_embedding["article_id"]})
        
        # Left join the recommendation counts with the full catalog.
        full_counts = full_catalog.join(rec_counts, on="article_id", how="left").fill_null(0)
        
        # Ensure the count column is numeric.
        full_counts = full_counts.with_columns(pl.col("count").cast(pl.Int64))
        
        # Sort counts in ascending order (required for the Gini calculation).
        full_counts = full_counts.sort("count")
        
        counts = np.array(full_counts["count"].to_list(), dtype=np.float64)
        n = len(counts)
        if n == 0 or np.sum(counts) == 0:
            return 0.0

        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * counts)) / (n * np.sum(counts))
        return gini


