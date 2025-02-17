import polars as pl
import numpy as np

def process_behavior_data(train_df: pl.DataFrame, test_df: pl.DataFrame) -> pl.DataFrame:
    """
    Process training and testing behavior data by exploding the "article_ids_clicked" column
    and filtering out rows where "article_ids_clicked" or "article_id" is null.
    
    Args:
        train_df (pl.DataFrame): The training behavior DataFrame.
        test_df (pl.DataFrame): The testing behavior DataFrame.
    
    Returns:
        pl.DataFrame: A combined DataFrame with processed behaviors.
    """
    # Processes training data
    processed_train_df = train_df.explode("article_ids_clicked")
    processed_train_df = processed_train_df.filter(pl.col("article_ids_clicked").is_not_null())
    processed_train_df = processed_train_df.filter(pl.col("article_id").is_not_null())

    # Processes testing data
    processed_test_df = test_df.explode("article_ids_clicked")
    processed_test_df = processed_test_df.filter(pl.col("article_ids_clicked").is_not_null())
    processed_test_df = processed_test_df.filter(pl.col("article_id").is_not_null())

    # Concatenates processed data
    combined_df = pl.concat([processed_train_df, processed_test_df])
    return combined_df

def random_split(df: pl.DataFrame, test_ratio: float = 0.30) -> (pl.DataFrame, pl.DataFrame):
    """
    Randomly split the DataFrame into training and test sets.
    
    Args:
        df (pl.DataFrame): The combined DataFrame.
        test_ratio (float, optional): Proportion of rows to use for the test set (default 0.30).
    
    Returns:
        tuple: A tuple (train_df, test_df) where train_df contains ~70% of the data
               and test_df contains ~30% of the data.
    """
    n = df.height
    test_mask = np.random.rand(n) < test_ratio
    test_df = df.filter(test_mask)
    train_df = df.filter(~test_mask)
    return train_df, test_df

def time_based_split(df: pl.DataFrame, test_ratio: float = 0.30) -> (pl.DataFrame, pl.DataFrame):
    """
    Split the DataFrame based on time. The oldest interactions (test_ratio percent)
    are used for testing, and the newest interactions are used for training.
    
    The DataFrame is sorted by "impression_time" and a secondary key ("impression_id")
    for stability in case of ties.
    
    Args:
        df (pl.DataFrame): The combined DataFrame.
        test_ratio (float, optional): Proportion of rows (oldest) to use for the test set (default 0.30).
    
    Returns:
        tuple: A tuple (train_df, test_df) where test_df contains the oldest interactions.
    """
    df_sorted = df.sort(["impression_time", "impression_id"])
    n_total = df_sorted.height
    n_test = int(n_total * test_ratio)
    test_df = df_sorted.head(n_test)
    train_df = df_sorted.tail(n_total - n_test)
    return train_df, test_df
