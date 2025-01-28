import polars as pl
import numpy as np

class CollaborativeReccomender:
    def __init__(self, user_item_df: pl.DataFrame):
        '''
        Initialize the CollaborativeReccomender with a user-item dataframe.

        Parameters
        ----------
        user_item_df : pl.DataFrame
            A Polars DataFrame with a unique 'user_id', 'article_id' pair, as well as numeric features for each user-item pair.
        '''
        self.user_item_df = user_item_df
        self.feature_cols = None
        self.user_profiles = None
        self.item_profiles = None