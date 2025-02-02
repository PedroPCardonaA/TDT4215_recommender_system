import polars as pl
import numpy as np
from joblib import Parallel, delayed

class CosineSimilarityContentBased:
    """
    A simple class to compute content-based recommendations using cosine similarity.
    """

    def __init__(self, behavior_data: pl.DataFrame, item_data: pl.DataFrame):
        """
        Initialize the content-based model.

        Args:
            behavior_data (pl.DataFrame): DataFrame containing user-item interactions with scores.
            item_data (pl.DataFrame): DataFrame containing item data with document vectors.
        """
        self.behavior_data = behavior_data
        self.item_data = item_data
        self.user_ids = behavior_data["user_id"].unique().to_numpy()
        self.item_ids = item_data["article_id"].unique().to_numpy()
        self.item_vectors = {row[0]: np.array(row[1]) for row in item_data.iter_rows()}  # {article_id: document_vector}

    def get_user_vector(self, user_id):
        """
        Compute the user profile vector as a weighted average of rated item vectors.

        Args:
            user_id (int): The ID of the user.

        Returns:
            np.array: The user profile vector.
        """
        user_ratings = self.behavior_data.filter(pl.col("user_id") == user_id)

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
        Recommend top n items for a given user using cosine similarity.

        Args:
            user_id (int): The ID of the user.
            n (int): The number of items to recommend.

        Returns:
            List[int]: A list of recommended item IDs.
        """
        user_vector = self.get_user_vector(user_id)
        rated_items = set(self.behavior_data.filter(pl.col("user_id") == user_id)["article_id"].to_numpy())

        similarities = []
        for item_id, item_vector in self.item_vectors.items():
            if item_id not in rated_items:  
                similarity = np.dot(user_vector, item_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(item_vector))
                similarities.append((item_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in similarities[:n]]

    def score(self, user_id, item_id):
        """
        Get the cosine similarity score between the user profile and an item.

        Args:
            user_id (int): The user ID.
            item_id (int): The item ID.

        Returns:
            float: The computed similarity score.
        """
        if item_id not in self.item_vectors:
            raise ValueError(f"Item ID {item_id} not found in item data.")

        user_vector = self.get_user_vector(user_id)
        item_vector = self.item_vectors[item_id]

        return float(np.dot(user_vector, item_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(item_vector)))

import polars as pl
import numpy as np
from joblib import Parallel, delayed

def precision_at_k(recommended_items, relevant_items, k=5):
    """
    Compute Precision@K.
    """
    if not relevant_items:
        return 0.0
    recommended_at_k = recommended_items[:k]
    hits = sum(1 for item in recommended_at_k if item in relevant_items)
    return hits / k

def ndcg_at_k(recommended_items, relevant_items, k=5):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at K.
    """
    def dcg(scores):
        return sum((score / np.log2(idx + 2)) for idx, score in enumerate(scores))

    recommended_at_k = recommended_items[:k]
    gains = [1 if item in relevant_items else 0 for item in recommended_at_k]
    
    ideal_gains = sorted([1] * len(relevant_items) + [0] * (k - len(relevant_items)), reverse=True)

    actual_dcg = dcg(gains)
    ideal_dcg = dcg(ideal_gains[:k])

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def compute_user_metrics(recommender, user_id, test_data, k=5):
    """
    Compute MAP@K and NDCG@K for a single user.
    """
    relevant_items = set(test_data.filter(pl.col("user_id") == user_id)["article_id"].to_numpy())
    if not relevant_items:
        return None  # Skip users with no interactions

    recommended_items = recommender.recommend(user_id, n=k)

    precision = precision_at_k(recommended_items, relevant_items, k)
    ndcg = ndcg_at_k(recommended_items, relevant_items, k)

    return precision, ndcg

def evaluate_recommender(recommender, test_data, k=5, n_jobs=-1):
    """
    Evaluate the recommender using MAP@K and NDCG@K in parallel.
    """
    user_ids = test_data["user_id"].unique().to_numpy()

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_user_metrics)(recommender, user_id, test_data, k) for user_id in user_ids
    )

    # Filter out None values (users with no interactions)
    results = [res for res in results if res is not None]

    if not results:
        return {"MAP@K": 0.0, "NDCG@K": 0.0}

    map_scores, ndcg_scores = zip(*results)

    return {
        "MAP@K": np.mean(map_scores),
        "NDCG@K": np.mean(ndcg_scores),
    }
