import numpy as np
import polars as pl
from joblib import Parallel, delayed


class UserItemBiasRecommender:
    """
    Implements a baseline user–item bias recommender.

    This model pivots a long-format user–item interaction DataFrame into a wide matrix,
    computes a global mean rating, and derives user and item biases. These biases are then
    used to predict ratings and generate recommendations.
    """

    def __init__(self, user_item_df: pl.DataFrame):
        """
        Initialize the recommender using a long-format user–item interaction DataFrame.

        The constructor pivots the input DataFrame into a wide user–item matrix using mean
        aggregation and fills missing interactions with 0. The expected input DataFrame should
        include the following columns:
          - `user_id` (UInt32)
          - `article_id` (Int32)
          - `impression_time` (Datetime with microsecond precision)
          - `score` (Float64)

        Parameters
        ----------
        user_item_df : pl.DataFrame
            Long-format DataFrame containing user–item interactions.

        Attributes
        ----------
        user_ids : list
            List of user IDs.
        item_ids : list
            List of item IDs.
        user_id_to_index : dict
            Mapping from user ID to its corresponding matrix row index.
        user_item_matrix : np.ndarray
            Wide user–item matrix with ratings as float32.
        global_mean : float
            Global mean rating computed during fitting.
        user_biases : np.ndarray or None
            Array of user biases.
        item_biases : np.ndarray or None
            Array of item biases.
        """
        wide_df = user_item_df.pivot(
            values="score",
            index="user_id",
            columns="article_id",
            aggregate_function="mean"
        )
        wide_df = wide_df.fill_null(0)
        self.user_ids = wide_df["user_id"].to_list()
        self.item_ids = [int(col) for col in wide_df.columns if col != "user_id"]
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.user_item_matrix = wide_df.select(
            [str(col) for col in self.item_ids]
        ).to_numpy().astype(np.float32)

        self.similarity_matrix = None
        self.global_mean = 0.0
        self.user_biases = None
        self.item_biases = None

    def fit(self):
        """
        Fit the model by computing the global mean, user biases, and item biases.

        The method calculates the global mean from non-zero ratings, then computes user biases
        as the average deviation of a user's ratings from the global mean. It then adjusts the
        item ratings by removing both the global mean and the user bias to compute item biases.
        """
        nonzero_ratings = self.user_item_matrix[self.user_item_matrix != 0]
        if nonzero_ratings.size > 0:
            self.global_mean = nonzero_ratings.mean()
        else:
            self.global_mean = 0.0

        num_users, num_items = self.user_item_matrix.shape
        self.user_biases = np.zeros(num_users, dtype=np.float32)
        self.item_biases = np.zeros(num_items, dtype=np.float32)
        for u in range(num_users):
            user_ratings = self.user_item_matrix[u]
            rated_idx = user_ratings != 0
            if np.any(rated_idx):
                self.user_biases[u] = (user_ratings[rated_idx] - self.global_mean).mean()

        for i in range(num_items):
            item_column = self.user_item_matrix[:, i]
            rated_idx = item_column != 0
            if np.any(rated_idx):
                self.item_biases[i] = (item_column[rated_idx]
                                       - self.global_mean
                                       - self.user_biases[rated_idx]).mean()

    def recommend(self, user_id, n=5):
        """
        Recommend the top-N items for a given user that have not been previously interacted with.

        The method predicts scores for all items that the user has not rated using:
            r̂(u, i) = global_mean + user_bias + item_bias
        and returns the items with the highest predicted scores.

        Parameters
        ----------
        user_id : int
            The identifier for the user.
        n : int, optional
            Number of items to recommend (default is 5).

        Returns
        -------
        list
            List of item IDs corresponding to the top-N recommendations.

        Raises
        ------
        ValueError
            If the model is not fitted or the `user_id` is not found.
        """
        if self.user_biases is None or self.item_biases is None:
            raise ValueError("The model must be fitted before making recommendations.")

        user_index = self.user_id_to_index.get(user_id)
        if user_index is None:
            raise ValueError(f"User ID {user_id} not found in the dataset.")

        user_interactions = self.user_item_matrix[user_index]
        unused_indices = np.where(user_interactions == 0)[0]

        scores = []
        for idx in unused_indices:
            predicted_score = (self.global_mean
                               + self.user_biases[user_index]
                               + self.item_biases[idx])
            scores.append((self.item_ids[idx], predicted_score))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_items = [item for item, _ in scores[:n]]
        return top_items

    def user_ratings(self, user_id):
        """
        Retrieve all ratings (interactions) for a specified user.

        Parameters
        ----------
        user_id : int
            The identifier for the user.

        Returns
        -------
        np.ndarray
            Array of ratings corresponding to the user.

        Raises
        ------
        ValueError
            If the user is not found in the dataset.
        """
        user_index = self.user_id_to_index.get(user_id)
        if user_index is None:
            raise ValueError(f"User ID {user_id} not found in the dataset.")
        return self.user_item_matrix[user_index]

    def predict(self, user_id, item_id):
        """
        Predict the rating for a given user–item pair.

        The prediction is computed as:
            r̂(u, i) = global_mean + user_bias + item_bias

        Parameters
        ----------
        user_id : int
            The identifier for the user.
        item_id : int
            The identifier for the item.

        Returns
        -------
        float
            The predicted rating.

        Raises
        ------
        ValueError
            If the model is not fitted or if the `user_id`/`item_id` is not found.
        """
        if self.user_biases is None or self.item_biases is None:
            raise ValueError("The model must be fitted before making predictions.")

        user_index = self.user_id_to_index.get(user_id)
        if user_index is None:
            raise ValueError(f"User ID {user_id} not found in the dataset.")
        if item_id not in self.item_ids:
            raise ValueError(f"Item ID {item_id} not found in the dataset.")

        item_index = self.item_ids.index(item_id)
        return float(
            self.global_mean
            + self.user_biases[user_index]
            + self.item_biases[item_index]
        )

    def precision_at_k(self, recommended_items, relevant_items, k=5):
        """
        Compute the Precision@K for a set of recommendations.

        Precision@K is the proportion of the top K recommended items that are relevant.

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
            The Precision@K value.
        """
        if not relevant_items:
            return 0.0
        recommended_at_k = recommended_items[:k]
        hits = sum(1 for item in recommended_at_k if item in relevant_items)
        return hits / k

    def ndcg_at_k(self, recommended_items, relevant_items, k=5):
        """
        Compute the Normalized Discounted Cumulative Gain (NDCG) at K.

        NDCG@K assesses ranking quality by measuring the position of relevant items in the top K.

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
            The NDCG@K value.
        """
        def dcg(scores):
            return sum(score / np.log2(idx + 2) for idx, score in enumerate(scores))

        recommended_at_k = recommended_items[:k]
        gains = [1 if item in relevant_items else 0 for item in recommended_at_k]
        ideal_gains = sorted([1] * len(relevant_items) + [0] * (k - len(relevant_items)), reverse=True)
        actual_dcg = dcg(gains)
        ideal_dcg = dcg(ideal_gains[:k])
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def compute_user_metrics(self, user_id, test_data: pl.DataFrame, k=5):
        """
        Compute Precision@K and NDCG@K metrics for a single user based on test interactions.

        The method filters the test data for the specified user, generates recommendations,
        and calculates both metrics.

        Parameters
        ----------
        user_id : int
            The identifier for the user.
        test_data : pl.DataFrame
            Long-format DataFrame with at least the columns "user_id" and "article_id".
        k : int, optional
            Number of top recommendations to consider (default is 5).

        Returns
        -------
        tuple or None
            A tuple (precision, ndcg) if relevant interactions exist; otherwise, None.
        """
        relevant_items = set(test_data.filter(pl.col("user_id") == user_id)["article_id"].to_numpy())
        if not relevant_items:
            return None

        recommended_items = self.recommend(user_id, n=k)
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        ndcg = self.ndcg_at_k(recommended_items, relevant_items, k)
        return precision, ndcg

    def evaluate_recommender(self, test_data: pl.DataFrame, k=5, n_jobs=-1, user_sample=None):
        """
        Evaluate the recommender's performance across multiple users in parallel.

        The method filters test users to those present in the training set, optionally samples
        a subset of users, and aggregates Precision@K and NDCG@K metrics.

        Parameters
        ----------
        test_data : pl.DataFrame
            Long-format DataFrame with columns "user_id" and "article_id".
        k : int, optional
            Number of top recommendations to consider per user (default is 5).
        n_jobs : int, optional
            Number of parallel jobs to run (default is -1 to use all available processors).
        user_sample : int, optional
            Number of users to sample for evaluation (if None, all eligible users are evaluated).

        Returns
        -------
        dict
            Dictionary containing average "Precision@K" and "NDCG@K" scores.
        """
        # Extract unique user IDs from the test data.
        user_ids = test_data["user_id"].unique().to_numpy()
        # Filter user IDs to retain only those present in the model.
        user_ids = np.array([u for u in user_ids if u in self.user_id_to_index])

        if user_sample is not None and user_sample < len(user_ids):
            user_ids = np.random.choice(user_ids, size=user_sample, replace=False)

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.compute_user_metrics)(user_id, test_data, k) for user_id in user_ids
        )

        # Exclude users with no test interactions.
        results = [res for res in results if res is not None]
        if not results:
            return {"Precision@K": 0.0, "NDCG@K": 0.0}

        precisions, ndcgs = zip(*results)
        return {"Precision@K": np.mean(precisions), "NDCG@K": np.mean(ndcgs)}

    def aggregate_diversity(self, item_df, k=5, user_sample=None, random_seed=42):
        """
        Compute the aggregate diversity (catalog coverage) of the recommendations.

        The metric measures the fraction of the total catalog that has been recommended
        across all users.

        Parameters
        ----------
        item_df : pl.DataFrame
            DataFrame containing at least the "article_id" column (the full catalog of items).
        k : int, optional
            Number of top recommendations per user (default is 5).
        user_sample : int, optional
            Number of users to sample for evaluation (if None, all users are evaluated).
        random_seed : int, optional
            Seed for reproducibility when sampling users (default is 42).

        Returns
        -------
        float
            The aggregate diversity as the fraction of the catalog that is recommended.
        """
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