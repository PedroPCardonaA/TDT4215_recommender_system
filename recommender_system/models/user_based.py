import polars as pl
import numpy as np

class CollaborativeReccomender:
    def __init__(self, impressions: pl.DataFrame, items: pl.DataFrame, scroll_percentage_weight = 1, read_time_weight = 1):
        '''
        Initialize the CollaborativeReccomender with a user-item dataframe.

        Parameters
        ----------
        TODO fix
        '''
        self.impressions = impressions
        self.items = items
        self.cosine_vector = None
        
        # Scoring weights 
        # TODO Perhaps add articles_clicked?
        # TODO Tweak for best results
        self.scroll_percentage_weight = scroll_percentage_weight
        self.read_time_weight = read_time_weight


    def calculate_impression_scores(self) -> pl.DataFrame:
        '''
        Creates a score based on the user's behavior (scroll percentage and read time).
        The formula is:
        impression_score = (scroll_percentage * scroll_percentage_weight) + (read_time * read_time_weight)

        Returns
        -------
        pl.DataFrame
            A DataFrame with an additional column `impression_score`.
        '''
        # Calculate the impression score using the given weights
        updated_impressions = self.impressions.with_columns(
            (
                pl.col("scroll_percentage") * self.scroll_percentage_weight +
                pl.col("read_time") * self.read_time_weight
            ).alias("impression_score")
        )
        return updated_impressions