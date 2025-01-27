import polars as pl
import numpy as np

class CosineSimilarityRecommender:
    def __init__(self, user_item_df: pl.DataFrame):
        self.user_item_df = user_item_df
        self.user_profiles = None
        self.item_profiles = None
        self.embedding_cols = None

    def fit(self):
        self.embedding_cols = [
            c for c in self.user_item_df.columns if c.startswith("vector_")
        ]
        self.user_profiles = (
            self.user_item_df
            .group_by("user_id")
            .agg([
                pl.col(c).mean().alias(c) for c in self.embedding_cols
            ])
        )
        self.item_profiles = (
            self.user_item_df
            .select(["article_id"] + self.embedding_cols)
            .unique(subset=["article_id"])
        )

        def norm_expr(cols):
            return (sum(pl.col(c)**2 for c in cols)).sqrt()

        self.user_profiles = self.user_profiles.with_columns(
            norm_expr(self.embedding_cols).alias("user_norm")
        )
        self.item_profiles = self.item_profiles.with_columns(
            norm_expr(self.embedding_cols).alias("item_norm")
        )

    def user_ratings(self, user_id: int) -> pl.DataFrame:
        return self.user_item_df.filter(pl.col("user_id") == user_id)

    def predict(self, user_id: int, article_id: int) -> float:
        if self.user_profiles is None or self.item_profiles is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        user_row = self.user_profiles.filter(pl.col("user_id") == user_id)
        if user_row.is_empty():
            return 0.0

        item_row = self.item_profiles.filter(pl.col("article_id") == article_id)
        if item_row.is_empty():
            return 0.0

        user_vec = user_row.select(self.embedding_cols).to_numpy()[0]
        item_vec = item_row.select(self.embedding_cols).to_numpy()[0]
        dot = float(np.dot(user_vec, item_vec))

        norm_u = float(user_row["user_norm"][0])
        norm_i = float(item_row["item_norm"][0])
        if norm_u == 0 or norm_i == 0:
            return 0.0

        return dot / (norm_u * norm_i)

    def recommend(self, user_id: int, n: int = 5) -> pl.DataFrame:
        if self.user_profiles is None or self.item_profiles is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        user_row = self.user_profiles.filter(pl.col("user_id") == user_id)
        if user_row.is_empty():
            return pl.DataFrame({"article_id": [], "score": []})
        rename_user = {}
        for c in user_row.columns:
            if c != "user_id":
                rename_user[c] = f"{c}_user"
        user_renamed = user_row.rename(rename_user)
        rename_item = {}
        for c in self.item_profiles.columns:
            rename_item[c] = f"{c}_item"  

        item_renamed = self.item_profiles.rename(rename_item)
        pairs = user_renamed.join(item_renamed, how="cross")

        dot_expr = sum(
            pl.col(f"{c}_user") * pl.col(f"{c}_item")
            for c in self.embedding_cols
        )

        pairs = pairs.with_columns(
            (
                dot_expr
                    / (pl.col("user_norm_user") * pl.col("item_norm_item"))
            ).alias("score")
        )
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
