import polars as pl
from typing import List, Any
import datetime
import numpy as np
import matplotlib.pyplot as plt


class RingBuffer:
    """
    Implements a simple ring buffer.

    This class maintains a fixed-size buffer that overwrites old items when full.
    """

    def __init__(self, size: int):
        """
        Initialize the ring buffer with a specified size.

        Parameters
        ----------
        size : int
            The fixed size of the ring buffer.
        """
        self.size = size
        self.buffer = [None] * size
        self.index = 0

    def append(self, item: Any) -> None:
        """
        Append an item to the ring buffer.

        The item is stored at the current index, and the index is advanced in a cyclic manner.

        Parameters
        ----------
        item : Any
            The item to append.
        """
        self.buffer[self.index] = item
        self.index = (self.index + 1) % self.size

    def get(self) -> List[Any]:
        """
        Retrieve all non-None items from the ring buffer.

        Returns
        -------
        List[Any]
            A list containing the stored items.
        """
        return [item for item in self.buffer if item is not None]

    def get_by_index(self, index: int) -> Any:
        """
        Retrieve the item at a specified index.

        Parameters
        ----------
        index : int
            The index from which to retrieve the item.

        Returns
        -------
        Any
            The item at the specified index.
        """
        return self.buffer[index]

    def get_most_popular(self, n: int) -> List[Any]:
        """
        Retrieve the N most popular items in the ring buffer.

        Popularity is determined by counting occurrences of each article ID.
        It assumes that each stored item is a list or tuple where the article ID is at index 1.

        Parameters
        ----------
        n : int
            Number of most popular items to retrieve.

        Returns
        -------
        List[Any]
            A list of the N most popular article IDs.
        """
        items = self.get()
        article_counts = {}
        for item in items:
            # Ensure the article ID is valid before counting.
            article_id = item[1]
            if article_id is None:
                continue
            article_counts[article_id] = article_counts.get(article_id, 0) + 1
        # Return the top n article IDs sorted by frequency.
        return sorted(article_counts, key=article_counts.get, reverse=True)[:n]

    def clear(self) -> None:
        """
        Clear the ring buffer by resetting all entries to None and the index to zero.
        """
        self.buffer = [None] * self.size
        self.index = 0

    def get_most_recent(self, n: int) -> List[Any]:
        """
        Retrieve the most recent N items from the ring buffer.

        Parameters
        ----------
        n : int
            The number of recent items to retrieve.

        Returns
        -------
        List[Any]
            A list containing the most recent N items. If there are fewer than N items,
            returns all available items.
        """
        items = self.get()
        return items[-n:] if len(items) >= n else items

    def __iter__(self):
        """
        Return an iterator over the items in the ring buffer.

        Returns
        -------
        iterator
            An iterator over the non-None items in the ring buffer.
        """
        return iter(self.get())

    def __len__(self) -> int:
        """
        Return the number of non-None items stored in the ring buffer.

        Returns
        -------
        int
            The count of stored items.
        """
        return len(self.get())


class RingBufferBaseline:
    """
    Implements a simple recommender system using a ring buffer to store the last N behaviors.

    This approach leverages both recency and popularity to recommend news articles.
    """

    def __init__(self, n: int = 10, behaviors: pl.DataFrame = None):
        """
        Initialize the recommender system with a ring buffer and behavior data.

        Parameters
        ----------
        n : int, optional
            The size of the ring buffer (default is 10).
        behaviors : pl.DataFrame, optional
            A DataFrame containing user behavior data. If provided, the DataFrame is sorted by
            "impression_time" in descending order.
        """
        self.n = n
        self.ring_buffer = RingBuffer(n)
        if behaviors is not None:
            # Sort provided behaviors by impression time in descending order.
            self.behaviors = behaviors.sort("impression_time", descending=True)
        else:
            self.behaviors = pl.DataFrame()

    def fit(self) -> None:
        """
        Fit the recommender by populating the ring buffer with all behavior entries.

        Each row from the behaviors DataFrame is appended to the ring buffer.
        """
        for i in range(len(self.behaviors)):
            # Append each behavior row to the ring buffer.
            self.ring_buffer.append(self.behaviors.row(i))

    def add_behavior(self, new_behavior: List[Any]) -> None:
        """
        Add a new behavior to the system.

        The new behavior is added to both the behaviors DataFrame and the ring buffer.

        Parameters
        ----------
        new_behavior : List[Any]
            A list representing a new behavior entry.
        """
        # Append the new behavior to the ring buffer.
        self.ring_buffer.append(new_behavior)
        # Add the new behavior to the behaviors DataFrame.
        if self.behaviors.shape[0] > 0:
            new_df = pl.DataFrame([new_behavior], schema=self.behaviors.columns)
            self.behaviors = self.behaviors.vstack(new_df)
        else:
            self.behaviors = pl.DataFrame([new_behavior])

    def recommend(self, user_id: int, n: int = 5) -> List[Any]:
        """
        Recommend the top N articles for a given user.

        The recommendation process starts from the most recent item in the ring buffer and
        walks backward until it finds articles different from the one the user is currently viewing.

        Parameters
        ----------
        user_id : int
            The user ID.
        n : int, optional
            The number of articles to recommend (default is 5).

        Returns
        -------
        List[Any]
            A list of recommended article IDs.
        """
        # Filter behaviors for the specified user.
        user_behaviors = self.behaviors.filter(pl.col("user_id") == user_id)
        if len(user_behaviors) == 0:
            # If the user has no behavior, recommend the most popular items from the ring buffer.
            recommendations = self.ring_buffer.get_most_popular(n)
            return recommendations

        # Determine the current article being viewed by the user (most recent behavior).
        current_article = (
            user_behaviors.sort("impression_time", descending=True)
                          .select("article_id")
                          .row(0)[0]
        )

        recommended_articles = []
        ring_size = self.ring_buffer.size

        # Iterate backwards through the ring buffer starting from the most recent entry.
        for i in range(ring_size):
            pos = (self.ring_buffer.index - 1 - i) % ring_size
            article = self.ring_buffer.get_by_index(pos)
            if article is not None:
                # Skip the article if it is the one the user is currently viewing.
                if article[1] == current_article:
                    continue
                recommended_articles.append(article)
                if len(recommended_articles) == n:
                    break

        # Return only the article IDs (assumed to be at index 1 in the stored item).
        return [article[1] for article in recommended_articles]

    def evaluate(self, test_data: pl.DataFrame, k: int = 5) -> dict:
        """
        Evaluate the recommender using precision, recall, and false positive rate at k.

        For each user in the test set, the set of relevant article IDs is compared with the recommended
        items. The candidate set for negatives is defined as all unique article IDs in the test data.

        Parameters
        ----------
        test_data : pl.DataFrame
            A DataFrame containing test interactions with columns "user_id" and "article_id".
        k : int, optional
            The number of top recommendations to consider (default is 5).

        Returns
        -------
        dict
            A dictionary with average "precision", "recall", and "fpr" (false positive rate).
        """
        # Build the candidate set from all unique article IDs in test_data.
        candidate_set = set(test_data.select("article_id").unique().to_numpy().flatten())

        # Retrieve unique user IDs from test_data.
        user_ids = test_data.select("user_id").unique().to_numpy().flatten()
        precisions = []
        recalls = []
        fprs = []

        for user in user_ids:
            # Get relevant items for this user.
            user_test = test_data.filter(pl.col("user_id") == user)
            relevant_items = set(user_test.select("article_id").to_numpy().flatten())
            if not relevant_items:
                continue

            # Obtain recommendations for the user.
            recommended_items = self.recommend(user, n=k)
            hits = sum(1 for item in recommended_items if item in relevant_items)
            precision = hits / k
            recall = hits / len(relevant_items)

            # Compute false positive rate.
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

    def reset_buffer(self) -> None:
        """
        Reset the ring buffer by clearing all stored behaviors.
        """
        self.ring_buffer.clear()