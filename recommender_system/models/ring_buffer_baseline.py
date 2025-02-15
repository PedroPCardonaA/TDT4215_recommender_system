import polars as pl
from typing import List, Any
import datetime
import numpy as np

class RingBuffer:
    """
    A simple ring buffer implementation.
    """
    def __init__(self, size: int):
        """
        Initialize the ring buffer with a given size.
        
        Args:
            size (int): The size of the ring buffer.
        """
        self.size = size
        self.buffer = [None] * size
        self.index = 0

    def append(self, item: Any):
        """
        Append an item to the ring buffer.
        
        Args:
            item: The item to append.
        """
        self.buffer[self.index] = item
        self.index = (self.index + 1) % self.size

    def get(self) -> List[Any]:
        """
        Get the items in the ring buffer.
        
        Returns:
            List: The items in the ring buffer.
        """
        return [item for item in self.buffer if item is not None]

    def get_by_index(self, index: int) -> Any:
        """
        Get the item at a specific index in the ring buffer.
        
        Args:
            index (int): The index of the item.
        
        Returns:
            Any: The item at the specified index.
        """
        return self.buffer[index]

    def get_most_popular(self, n: int) -> List[Any]:
        """
        Get the N most popular items in the ring buffer.
        """
        items = self.get()
        article_counts = {}
        for item in items:
            # Check if the article_id is valid before counting.
            article_id = item[1]
            if article_id is None:
                continue
            article_counts[article_id] = article_counts.get(article_id, 0) + 1
        return sorted(article_counts, key=article_counts.get, reverse=True)[:n]


    def clear(self):
        """
        Clears the ring buffer.
        """
        self.buffer = [None] * self.size
        self.index = 0

    def get_most_recent(self, n: int) -> List[Any]:
        """
        Retrieves the most recent N items from the ring buffer.
        
        Args:
            n (int): The number of recent items to retrieve.
        
        Returns:
            List: The most recent N items.
        """
        items = self.get()
        return items[-n:] if len(items) >= n else items

    def __iter__(self):
        """
        Get an iterator over the items in the ring buffer.
        
        Returns:
            Iterator: An iterator over the items in the ring buffer.
        """
        return iter(self.get())

    def __len__(self) -> int:
        """
        Get the number of items in the ring buffer.
        
        Returns:
            int: The number of items in the ring buffer.
        """
        return len(self.get())

class RingBufferBaseline:
    """
    Implements a simple recommender system that uses a ring buffer to store the last N items.
    This approach enforces both recency and popularity when recommending news articles.
    """
    def __init__(self, n: int = 10, behaviors: pl.DataFrame = None):
        """
        Initializes the recommender system with a ring buffer of size N.
        
        Args:
            n (int): The size of the ring buffer.
            behaviors (pl.DataFrame): A DataFrame containing user behavior data.
        """
        self.n = n
        self.ring_buffer = RingBuffer(n)
        if behaviors is not None:
            # Utilizes provided behaviors and sorts them by impression time in descending order.
            self.behaviors = behaviors.sort("impression_time", descending=True)
        else:
            self.behaviors = pl.DataFrame()

    def fit(self):
        """
        Fit the recommender system by adding all articles to the ring buffer.
        """
        for i in range(len(self.behaviors)):
            # Appends each row to the ring buffer.
            self.ring_buffer.append(self.behaviors.row(i))

    def add_behavior(self, new_behavior: List[Any]):
        """
        Adds a new behavior to both the behaviors DataFrame and the ring buffer.
        
        Args:
            new_behavior (List[Any]): A list representing a new behavior entry.
        """
        # Appends the new behavior to the ring buffer.
        self.ring_buffer.append(new_behavior)
        # Concatenates the new behavior to the behaviors DataFrame.
        if self.behaviors.shape[0] > 0:
            new_df = pl.DataFrame([new_behavior], schema=self.behaviors.columns)
            self.behaviors = self.behaviors.vstack(new_df)
        else:
            self.behaviors = pl.DataFrame([new_behavior])

    def recommend(self, user_id: int, n: int = 5) -> List[Any]:
        """
        Recommend the top N articles for a given user.
        The recommendation logic starts at the most recent item in the ring buffer
        (position index - 1) and walks backwards until it finds articles that are different
        from the article the user is currently viewing.
        
        Args:
            user_id (int): The user ID.
            n (int): The number of articles to recommend.
        
        Returns:
            List: The top N recommended article ids.
        """
        # Filters behaviors for the specified user.
        user_behaviors = self.behaviors.filter(pl.col("user_id") == user_id)
        #print("User behaviors for user", user_id, ":\n", user_behaviors) # DEBUG
        
        if len(user_behaviors) == 0:
            recommendations = self.ring_buffer.get_most_popular(n)
            return recommendations
        
        # Determine the article the user is currently viewing (most recent behavior).
        current_article = user_behaviors.sort("impression_time", descending=True) \
                                        .select("article_id").row(0)[0]
        
        recommended_articles = []
        ring_size = self.ring_buffer.size

        # Walk backwards through the ring buffer from the most recent insertion.
        for i in range(ring_size):
            pos = (self.ring_buffer.index - 1 - i) % ring_size
            article = self.ring_buffer.get_by_index(pos)
            # Only consider non-None articles.
            if article is not None:
                # Skip if this article is the one the user is currently viewing.
                if article[1] == current_article:
                    continue
                recommended_articles.append(article)
                if len(recommended_articles) == n:
                    break

        # Return only the article ids (assumed to be at index 1)
        return [article[1] for article in recommended_articles]
    

    def evaluate(self, test_data: pl.DataFrame, k: int = 5) -> dict:
        """
        Evaluate the recommender using precision, recall, and FPR at k.

        For each user in the test set, relevant items are defined as the set of article_ids
        the user has in the test data. The recommender's recommendations are then compared against
        these relevant items. The candidate set for negatives is defined as all article_ids in test_data.

        Args:
            test_data (pl.DataFrame): A DataFrame containing test interactions (with "user_id" and "article_id").
            k (int): The number of top recommendations to consider.

        Returns:
            dict: A dictionary with average precision, recall, and FPR.
        """
        # Candidate set: all unique article IDs in test_data.
        candidate_set = set(test_data.select("article_id").unique().to_numpy().flatten())
        
        # Get unique users from test data.
        user_ids = test_data.select("user_id").unique().to_numpy().flatten()
        precisions = []
        recalls = []
        fprs = []
        
        for user in user_ids:
            # Relevant items for this user.
            user_test = test_data.filter(pl.col("user_id") == user)
            relevant_items = set(user_test.select("article_id").to_numpy().flatten())
            if not relevant_items:
                continue

            recommended_items = self.recommend(user, n=k)  # list of recommended article ids

            # Compute hits.
            hits = sum(1 for item in recommended_items if item in relevant_items)
            precision = hits / k
            recall = hits / len(relevant_items)
            
            # Compute FPR:
            # Negatives: candidate_set minus relevant_items.
            negatives = candidate_set - relevant_items
            false_positives = sum(1 for item in recommended_items if item not in relevant_items)
            fpr = false_positives / len(negatives) if negatives else 0.0

            precisions.append(precision)
            recalls.append(recall)
            fprs.append(fpr)

        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_fpr = np.mean(fprs) if fprs else 0.0
        
        return {"precision": avg_precision, "recall": avg_recall, "fpr": avg_fpr}

    def roc_curve(self, test_data: pl.DataFrame, max_k: int = 10) -> dict:
        """
        Compute ROC curve coordinates (FPR and TPR) for thresholds from 1 to max_k.
        
        For each cutoff k, for each user the True Positive Rate (TPR) is the recall and
        the False Positive Rate (FPR) is computed based on the candidate set (all article_ids in test_data).
        The final ROC curve is obtained by averaging over all users.
        
        Args:
            test_data (pl.DataFrame): Test interactions DataFrame.
            max_k (int): Maximum cutoff value to consider.
        
        Returns:
            dict: A dictionary with keys "fpr" and "tpr", each a list of values for cutoffs 1...max_k.
        """
        candidate_set = set(test_data.select("article_id").unique().to_numpy().flatten())
        user_ids = test_data.select("user_id").unique().to_numpy().flatten()
        roc_fpr = []
        roc_tpr = []
        
        for k in range(1, max_k + 1):
            user_fpr = []
            user_tpr = []
            for user in user_ids:
                user_test = test_data.filter(pl.col("user_id") == user)
                relevant_items = set(user_test.select("article_id").to_numpy().flatten())
                if not relevant_items:
                    continue

                recommended_items = self.recommend(user, n=k)
                # Compute TPR (recall).
                tp = sum(1 for item in recommended_items if item in relevant_items)
                tpr = tp / len(relevant_items)
                # Compute FPR.
                negatives = candidate_set - relevant_items
                fp = sum(1 for item in recommended_items if item not in relevant_items)
                fpr = fp / len(negatives) if negatives else 0.0

                user_tpr.append(tpr)
                user_fpr.append(fpr)
            # Average over users.
            roc_tpr.append(np.mean(user_tpr) if user_tpr else 0.0)
            roc_fpr.append(np.mean(user_fpr) if user_fpr else 0.0)
        
        return {"fpr": roc_fpr, "tpr": roc_tpr}


    def reset_buffer(self):
        """
        Resets the ring buffer, clearing all stored behaviors.
        """
        self.ring_buffer.clear()

def main(): # TESTING
    # Creates a sample dataset resembling the EB-NeRD news dataset.
    # The dataset includes columns: impression_id, article_id, impression_time, and user_id.
    data = {
        "impression_id": [1, 2, 3, 4, 5],
        "article_id": [100, 101, 102, 102, 104],
        "impression_time": [
            datetime.datetime(2025, 2, 14, 10, 0, 0),
            datetime.datetime(2025, 2, 14, 10, 1, 0),
            datetime.datetime(2025, 2, 14, 10, 2, 0),
            datetime.datetime(2025, 2, 14, 10, 3, 0),
            datetime.datetime(2025, 2, 14, 10, 4, 0)
        ],
        "user_id": [123, 456, 123, 789, 123]
    }
    # Creates a Polars DataFrame from the sample data.
    df = pl.DataFrame(data)
    print("Sample DataFrame:")
    print(df)

    # Initializes the recommender system with the sample DataFrame.
    recommender = RingBufferBaseline(behaviors=df)
    # Fits the recommender system (populates the ring buffer).
    recommender.fit()

    # Prints the content of the ring buffer.
    print("\nRing Buffer Content:")
    for item in recommender.ring_buffer.get():
         print(item)

    # Retrieves recommendations for a user who has behavior (user_id=123).
    recommendations = recommender.recommend(user_id=123, n=5)
    print("\nRecommended articles for user 123:")
    for rec in recommendations:
         print(rec)
         
    # Retrieves recommendations for a user with no behavior (user_id=999).
    recommendations_no = recommender.recommend(user_id=999, n=5)
    print("\nRecommended articles for user 999 (no behavior):")
    for rec in recommendations_no:
         print(rec)

# Executes main() if this script is run directly.
if __name__ == '__main__':
    main()
