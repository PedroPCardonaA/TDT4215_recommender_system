import polars as pl
import numpy as np

def process_behavior_data(train_df: pl.DataFrame, test_df: pl.DataFrame, relevant_columns=["impression_id", "article_id", "impression_time", "user_id"]) -> pl.DataFrame:
    """
    Process training and testing behavior data by selecting relevant columns,
    filtering out rows with null values, and sorting by "impression_time" in descending order.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training behavior DataFrame.
    test_df : pl.DataFrame
        Testing behavior DataFrame.
    relevant_columns : list, optional
        List of columns to keep (default is ["impression_id", "article_id", "impression_time", "user_id"]).

    Returns
    -------
    pl.DataFrame
        A combined DataFrame containing processed behavior data, sorted by "impression_time" in descending order.
    """

    # Keep only relevant columns.
    train_behaviors_df = train_df.select(relevant_columns)
    test_behaviors_df = test_df.select(relevant_columns)

    # Filter out rows with null-entries.
    processed_train_df = train_behaviors_df.filter(pl.col("article_id").is_not_null())
    processed_test_df = test_behaviors_df.filter(pl.col("article_id").is_not_null())

    # Sort the DataFrame by "impression_time" in descending order.
    processed_train_df = processed_train_df.sort("impression_time", descending=True)
    processed_test_df = processed_test_df.sort("impression_time", descending=True)

    # Concatenate the processed training and testing data.
    combined_df = pl.concat([processed_train_df, processed_test_df])
    return combined_df


def random_split(df: pl.DataFrame, test_ratio: float = 0.30) -> (pl.DataFrame, pl.DataFrame):
    """
    Randomly split a DataFrame into training and testing sets.

    Parameters
    ----------
    df : pl.DataFrame
        The combined DataFrame to split.
    test_ratio : float, optional
        Proportion of rows to use for the test set (default is 0.30).

    Returns
    -------
    tuple
        A tuple (train_df, test_df) where approximately (1 - test_ratio) of the rows form the
        training set and test_ratio of the rows form the test set.
    """
    n = df.height
    # Generate a boolean mask where True indicates a row goes to the test set.
    test_mask = np.random.rand(n) < test_ratio
    # Filter rows for test and training sets using the mask.
    test_df = df.filter(test_mask)
    train_df = df.filter(~test_mask)
    return train_df, test_df


def time_based_split(df: pl.DataFrame, test_ratio: float = 0.30) -> (pl.DataFrame, pl.DataFrame):
    """
    Split a DataFrame into training and test sets based on time.

    The method sorts the DataFrame by "impression_time" and "impression_id" (as a secondary key)
    to ensure a stable order, then uses the oldest interactions (test_ratio percent) as the test set.

    Parameters
    ----------
    df : pl.DataFrame
        The combined DataFrame to split.
    test_ratio : float, optional
        The proportion of rows (from the oldest interactions) to use for the test set (default is 0.30).

    Returns
    -------
    tuple
        A tuple (train_df, test_df) where test_df contains the oldest interactions and train_df
        contains the remainder.
    """
    # Sort the DataFrame by "impression_time" and "impression_id" for stability.
    df_sorted = df.sort(["impression_time", "impression_id"])
    n_total = df_sorted.height
    # Calculate the number of rows to allocate to the test set.
    n_test = int(n_total * test_ratio)
    # Select the oldest interactions for the test set.
    test_df = df_sorted.head(n_test)
    # Use the remaining rows for the training set.
    train_df = df_sorted.tail(n_total - n_test)
    return train_df, test_df
