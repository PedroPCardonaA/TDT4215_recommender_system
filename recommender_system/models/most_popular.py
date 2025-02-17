import polars as pl
from typing import List
import numpy as np

class MostPopularRecommender:
    """
    A simple popularity-based recommender system that recommends the most popular articles to any user.
    """
    def __init__(self, behaviors: pl.DataFrame):
        """
        Initializes the popularity-based recommender using the behaviors DataFrame.
        Expects the DataFrame to have a column "Clicked Article IDs" that contains lists of article IDs.
        """
        self.behaviors = behaviors
        self.top_articles = []

    def fit(self):
        """
        Computes the popularity of each article based on the frequency of clicks.
        Explodes the list of clicked article IDs and counts the number of occurrences per article.
        """
        popularity = self.behaviors.group_by("article_ids_clicked").agg(pl.count().alias("click_count"))
        popularity = popularity.sort("click_count", descending=True)
        # Stores the sorted list of article IDs
        self.top_articles = popularity["article_ids_clicked"]

    def recommend(self, user_id: int, n: int = 5) -> List[int]: 
        """
        Returns the top-k most popular articles for to any user,
        this means jsut the basic most popular articles based on the click count.
        """
        return self.top_articles[:n]