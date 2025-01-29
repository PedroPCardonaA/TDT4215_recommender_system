import polars as pl

def user_item_interaction_scores(behaviors: pl.DataFrame) -> pl.DataFrame:
    behaviors = behaviors.drop_nans(subset=["article_id", "user_id"])
    
    behaviors = behaviors[
        ["user_id", "article_id", "read_time", "scroll_percentage"]
    ]
    
    behaviors = behaviors.group_by(["user_id", "article_id"]).agg(
        pl.col("read_time").sum(),
        pl.col("scroll_percentage").max()
    )
    behaviors = behaviors.with_columns(
        (pl.col("read_time").add(1).log()).alias("read_time_log")
    )
    behaviors = behaviors.with_columns(
        (pl.col("read_time_log") * pl.col("scroll_percentage")).alias("score")
    )
    behaviors = behaviors.drop(["read_time", "scroll_percentage", "read_time_log"])
    behaviors = behaviors.with_columns(
        ((pl.col("score") - pl.col("score").min()) / (pl.col("score").max() - pl.col("score").min())).alias("score")
    )
    return behaviors
