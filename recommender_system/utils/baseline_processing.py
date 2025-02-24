import polars as pl
import numpy as np

def process_behavior_data(train_df: pl.DataFrame, test_df: pl.DataFrame) -> pl.DataFrame:
    """
    Process training and testing behavior data by exploding the "article_ids_clicked" column
    and filtering out rows with null values.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training behavior DataFrame.
    test_df : pl.DataFrame
        Testing behavior DataFrame.

    Returns
    -------
    pl.DataFrame
        A combined DataFrame containing processed behavior data.
    """
    # Process the training data by exploding the "article_ids_clicked" column
    processed_train_df = train_df.explode("article_ids_clicked")
    # Filter out rows with null "article_ids_clicked"
    processed_train_df = processed_train_df.filter(pl.col("article_ids_clicked").is_not_null())
    # Filter out rows with null "article_id"
    processed_train_df = processed_train_df.filter(pl.col("article_id").is_not_null())

    # Process the testing data similarly
    processed_test_df = test_df.explode("article_ids_clicked")
    processed_test_df = processed_test_df.filter(pl.col("article_ids_clicked").is_not_null())
    processed_test_df = processed_test_df.filter(pl.col("article_id").is_not_null())

    # Concatenate the processed training and testing data
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
