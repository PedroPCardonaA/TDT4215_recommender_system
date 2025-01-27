import polars as pl
import numpy as np

class CosineSimilarityRecommender:
    """
    A recommender system that computes average user feature vectors
    and unique item feature vectors, then predicts ratings using the
    cosine similarity between these vectors.

    This class expects a Polars DataFrame (`user_item_df`) containing:
        - 'user_id': unique identifier for the user
        - 'article_id': unique identifier for the item (or article)
        - Other columns representing numeric features (these can be embeddings
          or any numeric attributes).

    Attributes
    ----------
    user_item_df : pl.DataFrame
        The input dataframe containing user IDs, item IDs, and numeric features.

    feature_cols : list of str
        A list of column names representing all numeric features (excluding
        'user_id' and 'article_id').

    user_profiles : pl.DataFrame or None
        DataFrame containing the average feature vector for each user
        along with a precomputed norm (magnitude) of that vector. Created after
        calling `fit()`.

    item_profiles : pl.DataFrame or None
        DataFrame containing the feature vector for each item
        along with a precomputed norm (magnitude) of that vector. Created after
        calling `fit()`.

    Methods
    -------
    __init__(user_item_df: pl.DataFrame)
        Constructor method.

    fit()
        Aggregates user vectors (to get user profiles) and collects unique item vectors
        (to get item profiles), then computes their vector norms.

    user_ratings(user_id: int) -> pl.DataFrame
        Returns the rows from `user_item_df` for a specific user.

    predict(user_id: int, article_id: int) -> float
        Computes the cosine similarity between the user's mean feature vector
        and the item's feature vector.

    recommend(user_id: int, n: int = 5) -> pl.DataFrame
        Returns the top-n items for a given user, ranked by descending cosine similarity.
    """

    def __init__(self, user_item_df: pl.DataFrame):
        """
        Initialize the CosineSimilarityRecommender with a user-item dataframe.

        Parameters
        ----------
        user_item_df : pl.DataFrame
            A Polars DataFrame with columns:
            'user_id', 'article_id', and numeric features for each user-item pair.
        """
        self.user_item_df = user_item_df
        self.feature_cols = None
        self.user_profiles = None
        self.item_profiles = None

    def fit(self):
        """
        Prepares user and item profiles by computing average user feature vectors
        (user_profiles) and unique item feature vectors (item_profiles).
        Also calculates norms for each user and item vector to speed up
        cosine similarity.
        """
        # Identify numeric feature columns (excluding user_id & article_id)
        # Assumes all other columns are numeric
        self.feature_cols = [
            c for c in self.user_item_df.columns
            if c not in ("user_id", "article_id")
        ]

        # Compute the mean of each feature per user
        self.user_profiles = (
            self.user_item_df
            .group_by("user_id")
            .agg([
                pl.col(c).mean().alias(c) for c in self.feature_cols
            ])
        )

        # Extract unique items and their feature vectors
        self.item_profiles = (
            self.user_item_df
            .select(["article_id"] + self.feature_cols)
            .unique(subset=["article_id"])
        )

        # Helper function to compute vector norm
        def norm_expr(cols):
            # Expression to compute vector norm (sqrt of sum of squares)
            return (sum(pl.col(c)**2 for c in cols)).sqrt()

        # Compute and store the user norm
        self.user_profiles = self.user_profiles.with_columns(
            norm_expr(self.feature_cols).alias("user_norm")
        )

        # Compute and store the item norm
        self.item_profiles = self.item_profiles.with_columns(
            norm_expr(self.feature_cols).alias("item_norm")
        )

    def user_ratings(self, user_id: int) -> pl.DataFrame:
        """
        Retrieve all rows from the user-item DataFrame related to a specific user.

        Parameters
        ----------
        user_id : int
            The ID of the user for which we want to retrieve rows.

        Returns
        -------
        pl.DataFrame
            A Polars DataFrame containing all rows for the given user.
        """
        return self.user_item_df.filter(pl.col("user_id") == user_id)

    def predict(self, user_id: int, article_id: int) -> float:
        """
        Predict the similarity score (cosine similarity) between a user's mean
        feature vector and an item's feature vector.

        Parameters
        ----------
        user_id : int
            The ID of the user for whom we want a prediction.

        article_id : int
            The ID of the item (article) for which we want a prediction.

        Returns
        -------
        float
            A score representing the cosine similarity. Ranges from -1 to 1, but
            typically [0, 1] if features are non-negative. Returns 0.0 if the
            user or item is not found or if norms are zero (to avoid division by zero).

        Raises
        ------
        RuntimeError
            If the model is not fitted (no user/item profiles exist).
        """
        if self.user_profiles is None or self.item_profiles is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Retrieve user profile
        user_row = self.user_profiles.filter(pl.col("user_id") == user_id)
        if user_row.is_empty():
            return 0.0

        # Retrieve item profile
        item_row = self.item_profiles.filter(pl.col("article_id") == article_id)
        if item_row.is_empty():
            return 0.0

        # Extract numpy vectors
        user_vec = user_row.select(self.feature_cols).to_numpy()[0]
        item_vec = item_row.select(self.feature_cols).to_numpy()[0]
        dot = float(np.dot(user_vec, item_vec))

        # Retrieve norms
        norm_u = float(user_row["user_norm"][0])
        norm_i = float(item_row["item_norm"][0])

        if norm_u == 0 or norm_i == 0:
            return 0.0

        # Return cosine similarity
        return dot / (norm_u * norm_i)

    def recommend(self, user_id: int, n: int = 5) -> pl.DataFrame:
        """
        Recommend the top-n items for a given user based on cosine similarity
        between the user's mean feature vector and each item's feature vector.

        Parameters
        ----------
        user_id : int
            The user ID for whom we want to recommend items.
        n : int, optional
            Number of recommended items to return (default is 5).

        Returns
        -------
        pl.DataFrame
            A Polars DataFrame containing columns 'article_id' and 'score', where
            'score' is the cosine similarity. Sorted by descending similarity.

        Raises
        ------
        RuntimeError
            If the model is not fitted (no user/item profiles exist).
        """
        if self.user_profiles is None or self.item_profiles is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        user_row = self.user_profiles.filter(pl.col("user_id") == user_id)
        if user_row.is_empty():
            # If user doesn't exist, return an empty DataFrame
            return pl.DataFrame({"article_id": [], "score": []})

        # Rename columns to avoid name collisions during cross-join
        rename_user = {}
        for c in user_row.columns:
            if c != "user_id":
                rename_user[c] = f"{c}_user"
        user_renamed = user_row.rename(rename_user)

        rename_item = {}
        for c in self.item_profiles.columns:
            rename_item[c] = f"{c}_item"
        item_renamed = self.item_profiles.rename(rename_item)

        # Cross join to get every (user, item) pair for scoring
        pairs = user_renamed.join(item_renamed, how="cross")

        # Dot product of user vector and item vector
        dot_expr = sum(
            pl.col(f"{c}_user") * pl.col(f"{c}_item")
            for c in self.feature_cols
        )

        # Compute cosine similarity score
        pairs = pairs.with_columns(
            (
                dot_expr / (pl.col("user_norm_user") * pl.col("item_norm_item"))
            ).alias("score")
        )

        # Select needed columns, sort by similarity, and return top-n
        result = (
            pairs
            .select([
                pl.col("article_id_item").alias("article_id"),
                pl.col("score"),
            ])
            .sort("score", descending=True)
            .head(n)
        )
        return result
