import numpy as np
import polars as pl
from joblib import Parallel, delayed

class UserItemBiasRecommender:
    def __init__(self, user_item_df: pl.DataFrame):
        """
        Initialize the recommender system using a long-format user-item interaction dataframe.
        Expected schema:
            - user_id (UInt32)
            - article_id (Int32)
            - impression_time (Datetime with microsecond precision)
            - score (Float64)
            
        The constructor pivots the long dataframe into a wide user-item matrix (using mean aggregation)
        and fills missing interactions with 0.
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
        Compute global mean, then user biases, then item biases.
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
        Recommend the top-N items (by predicted score) that the user has not interacted with.
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
        Retrieve all ratings (interactions) for a given user.
        """
        user_index = self.user_id_to_index.get(user_id)
        if user_index is None:
            raise ValueError(f"User ID {user_id} not found in the dataset.")
        return self.user_item_matrix[user_index]

    def predict(self, user_id, item_id):
        """
        Predict the (implicit or explicit) rating for a given user and item using:
            r_hat(u, i) = mu + b_u + b_i
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
        Compute Precision@K for a given list of recommended items and a set of relevant items.
        """
        if not relevant_items:
            return 0.0
        recommended_at_k = recommended_items[:k]
        hits = sum(1 for item in recommended_at_k if item in relevant_items)
        return hits / k

    def ndcg_at_k(self, recommended_items, relevant_items, k=5):
        """
        Compute Normalized Discounted Cumulative Gain (NDCG) at K.
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
        Compute Precision@K and NDCG@K for a single user based on test interactions.
        The test_data should be a long-format dataframe with at least "user_id" and "article_id" columns.
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
        Evaluate the recommender using Precision@K and NDCG@K in parallel on a sample of users.
        test_data must be a long-format dataframe with columns "user_id" and "article_id".
        Only users that exist in the training set (self.user_ids) are considered.
        """
        # Get unique user IDs from the test data
        user_ids = test_data["user_id"].unique().to_numpy()
        # Filter user IDs to only include those present in the model
        user_ids = np.array([u for u in user_ids if u in self.user_id_to_index])
        
        if user_sample is not None and user_sample < len(user_ids):
            user_ids = np.random.choice(user_ids, size=user_sample, replace=False)
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.compute_user_metrics)(user_id, test_data, k) for user_id in user_ids
        )
        
        # Filter out users with no test interactions (or where compute_user_metrics returned None)
        results = [res for res in results if res is not None]
        if not results:
            return {"Precision@K": 0.0, "NDCG@K": 0.0}
        
        precisions, ndcgs = zip(*results)
        return {"Precision@K": np.mean(precisions), "NDCG@K": np.mean(ndcgs)}

