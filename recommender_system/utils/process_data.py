import polars as pl

def user_item_interaction_scores(behaviors: pl.DataFrame, article: pl.DataFrame) -> pl.DataFrame:
    behaviors = behaviors.drop_nans(subset=["article_id", "user_id"])
    
    behaviors = behaviors[
        ["user_id", "article_id", "read_time", "scroll_percentage","impression_time"]
    ]
    
    behaviors = behaviors.group_by(["user_id", "article_id"]).agg(
    pl.col("read_time").sum(),
    pl.col("scroll_percentage").max(),
    pl.col("impression_time").max().alias("impression_time")
)
    
    behaviors = behaviors.with_columns(
        (pl.col("read_time").add(1).log()).alias("read_time_log")
    )
    
    behaviors = behaviors.with_columns(
        (pl.col("read_time_log") * pl.col("scroll_percentage")).alias("score")
    )
    
    behaviors = behaviors.drop(["read_time", "scroll_percentage", "read_time_log"])
    
    # Normalize the score
    behaviors = behaviors.with_columns(
        ((pl.col("score") - pl.col("score").min()) / (pl.col("score").max() - pl.col("score").min())).alias("score")
    )
    
    # Convert published_time to numerical epoch value
    article = article.with_columns(
        pl.col("published_time").cast(pl.Datetime).dt.timestamp().alias("published_time_epoch")
    )
    
    # Normalize published_time
    article = article.with_columns(
        ((pl.col("published_time_epoch") - pl.col("published_time_epoch").min()) / (pl.col("published_time_epoch").max() - pl.col("published_time_epoch").min())).alias("published_time_norm")
    )
    
    # Join with articles dataframe
    behaviors = behaviors.join(article.select(["article_id", "published_time_norm"]), on="article_id", how="left")
    
    # Update score to include article freshness
    behaviors = behaviors.with_columns(
        (pl.col("score") * 0.8 + pl.col("published_time_norm") * 0.2).alias("score")
    )
    
    behaviors = behaviors.drop([ "published_time_norm"])
    
    return behaviors


def user_item_binary_interaction(
    behaviors: pl.DataFrame, users: pl.DataFrame, articles: pl.DataFrame
) -> pl.DataFrame:
    # Select only necessary columns
    users = users.select(["user_id"])
    articles = articles.select(["article_id"])
    
    # Create a cross join to generate all possible (user, article) pairs
    user_article_pairs = users.join(articles, how="cross")
    
    # Ensure behaviors has only relevant columns
    behaviors = behaviors.select(["user_id", "article_id"]).with_columns(pl.lit(1).alias("clicked"))
    
    # Left join user-article pairs with behaviors to mark interactions
    interaction_matrix = user_article_pairs.join(
        behaviors, on=["user_id", "article_id"], how="left"
    ).fill_null(0)
    
    return interaction_matrix