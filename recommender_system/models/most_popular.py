import polars as pl
from typing import List

class MostPopularRecommender:
    """
    Implements a popularity-based recommender system that suggests the most popular articles to any user.
    """

    def __init__(self, behaviors: pl.DataFrame):
        """
        Initialize the recommender with a user behaviors DataFrame.

        Parameters
        ----------
        behaviors : pl.DataFrame
            DataFrame containing user behavior data. It is expected to have a column
            'article_ids_clicked' that contains lists of article IDs clicked by users.
        """
        self.behaviors = behaviors
        self.top_articles: List[int] = []  # Stores the sorted list of popular article IDs.

    def fit(self) -> None:
        """
        Fit the recommender by computing article popularity based on click frequency.

        The method groups the behaviors by 'article_ids_clicked', counts the number of occurrences,
        sorts the articles by click count in descending order, and stores the sorted article IDs.
        """
        # Group by article_ids_clicked and count clicks.
        popularity = self.behaviors.group_by("article_ids_clicked").agg(pl.count().alias("click_count"))
        # Sort articles by click_count in descending order.
        popularity = popularity.sort("click_count", descending=True)
        # Store the sorted article IDs.
        self.top_articles = popularity["article_ids_clicked"]

    def recommend(self, user_id: int, n: int = 5) -> List[int]:
        """
        Recommend the top-n most popular articles.

        Parameters
        ----------
        user_id : int
            The user ID. This recommender returns the same popular articles for every user.
        n : int, optional
            Number of top articles to return (default is 5).

        Returns
        -------
        List[int]
            A list of the most popular article IDs.
        """
        return self.top_articles[:n]
