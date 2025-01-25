import numpy as np
import polars as pl

class UserItemBiasRecommender:
    def __init__(self, user_item_df: pl.DataFrame):
        """
        Initialize the recommender system with a user-item interaction matrix.
        """
        self.user_ids = user_item_df["user_id"].to_list()
        self.item_ids = [int(col) for col in user_item_df.columns if col != "user_id"]
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.user_item_matrix = user_item_df.select(
            [str(col) for col in self.item_ids]
        ).to_numpy()
        self.similarity_matrix = None
        self.global_mean = 0.0
        self.user_biases = None
        self.item_biases = None

    def fit(self):
        """
        Compute global mean, then user biases, then item biases.
        """
        nonzero_ratings = self.user_item_matrix[self.user_item_matrix != 0]
        if len(nonzero_ratings) > 0:
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
