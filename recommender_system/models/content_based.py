import polars as pl
import numpy as np
from joblib import Parallel, delayed

class CosineSimilarityContentBased:
    """
    A content-based recommender that splits user behavior into training and testing sets
    based on the impression_time column. The training set contains the first 80% of a user's
    interactions (chronologically) and the test set the remaining 20%.
    """

    def __init__(self, behavior_data: pl.DataFrame, item_data: pl.DataFrame, train_ratio: float = 0.8):
        """
        Initialize the model and split behavior data into training and testing sets.
        
        Args:
            behavior_data (pl.DataFrame): DataFrame with columns such as user_id, article_id,
                                          score, and impression_time.
            item_data (pl.DataFrame): DataFrame containing item data with document vectors.
            train_ratio (float): Fraction of interactions to use for training (default 0.8).
        """
        self.train_data, self.test_data = self.split_behavior_data(behavior_data, train_ratio, time_column="impression_time")
        self.item_data = item_data
        self.user_ids = self.train_data["user_id"].unique().to_numpy()
        self.item_ids = item_data["article_id"].unique().to_numpy()
        self.item_vectors = {row[0]: np.array(row[1]) for row in item_data.iter_rows()}  # {article_id: document_vector}

    def split_behavior_data(self, behavior_data: pl.DataFrame, train_ratio: float, time_column: str = "impression_time"):
        """
        Split each user's behavior data into training and testing sets based on the provided ratio.
        
        Args:
            behavior_data (pl.DataFrame): The full behavior data.
            train_ratio (float): Ratio of interactions to allocate to training.
            time_column (str): The column name indicating the time of interaction.
        
        Returns:
            tuple: (train_data, test_data) as two Polars DataFrames.
        """
        user_ids = behavior_data["user_id"].unique().to_numpy()
        train_data_list = []
        test_data_list = []
        
        for user_id in user_ids:
            user_data = behavior_data.filter(pl.col("user_id") == user_id).sort(time_column)
            n = user_data.height
            if n < 2:
                train_data_list.append(user_data)
            else:
                train_cutoff = int(n * train_ratio)
                if train_cutoff == n:
                    train_cutoff = n - 1
                train_data_list.append(user_data[:train_cutoff])
                test_data_list.append(user_data[train_cutoff:])
        
        train_data = pl.concat(train_data_list)
        test_data = pl.concat(test_data_list) if test_data_list else pl.DataFrame()
        return train_data, test_data

    def get_user_vector(self, user_id):
        """
        Compute the user profile vector as a weighted average of rated item vectors,
        using only the training data.
        
        Args:
            user_id (int): The ID of the user.
        
        Returns:
            np.array: The user profile vector.
        """
        user_ratings = self.train_data.filter(pl.col("user_id") == user_id)

        if user_ratings.is_empty():
            return np.mean(list(self.item_vectors.values()), axis=0)

        rated_items = user_ratings["article_id"].to_numpy()
        scores = user_ratings["score"].to_numpy()
        score_sum = np.sum(scores)

        if score_sum == 0:
            raise ValueError(f"User {user_id} has all zero ratings.")

        user_vector = np.zeros(len(next(iter(self.item_vectors.values()))))
        for item_id, score in zip(rated_items, scores):
            user_vector += score * self.item_vectors[item_id]

        user_vector /= score_sum
        return user_vector

    def recommend(self, user_id, n=5):
        """
        Recommend top n items for a given user based on cosine similarity.
        
        Args:
            user_id (int): The ID of the user.
            n (int): The number of items to recommend.
        
        Returns:
            List[int]: A list of recommended item IDs.
        """
        user_vector = self.get_user_vector(user_id)
        rated_items = set(self.train_data.filter(pl.col("user_id") == user_id)["article_id"].to_numpy())

        similarities = []
        for item_id, item_vector in self.item_vectors.items():
            if item_id not in rated_items:
                similarity = np.dot(user_vector, item_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(item_vector))
                similarities.append((item_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in similarities[:n]]

    def score(self, user_id, item_id):
        """
        Compute the cosine similarity score between the user profile (from training data)
        and an item.
        
        Args:
            user_id (int): The user ID.
            item_id (int): The item ID.
        
        Returns:
            float: The cosine similarity score.
        """
        if item_id not in self.item_vectors:
            raise ValueError(f"Item ID {item_id} not found in item data.")

        user_vector = self.get_user_vector(user_id)
        item_vector = self.item_vectors[item_id]

        return float(np.dot(user_vector, item_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(item_vector)))
    
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

    def compute_user_metrics(self, user_id, k=5):
        """
        Compute Precision@K and NDCG@K for a single user.
        
        Args:
            user_id (int): The user ID.
            k (int): Number of top recommendations to consider.
            
        Returns:
            tuple or None: (precision, ndcg) scores, or None if the user has no test interactions.
        """
        relevant_items = set(self.test_data.filter(pl.col("user_id") == user_id)["article_id"].to_numpy())
        if not relevant_items:
            return None 

        recommended_items = self.recommend(user_id, n=k)
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
        user_ids = self.test_data["user_id"].unique().to_numpy()

        if user_sample is not None and user_sample < len(user_ids):
            user_ids = np.random.choice(user_ids, size=user_sample, replace=False)
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.compute_user_metrics)(user_id, k) for user_id in user_ids
        )
        results = [res for res in results if res is not None]
        
        if not results:
            return {"MAP@K": 0.0, "NDCG@K": 0.0}
        
        map_scores, ndcg_scores = zip(*results)
        
        return {
            "MAP@K": np.mean(map_scores),
            "NDCG@K": np.mean(ndcg_scores),
        }
