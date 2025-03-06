import polars as pl
import numpy as np

class DataProcesser:
    articles_df: pl.DataFrame
    document_vectors_df: pl.DataFrame
    train_behaviors_df: pl.DataFrame
    test_behaviors_df: pl.DataFrame

    def __init__(self):
        # Load the data and store as instance variables
        self.articles_df = pl.read_parquet('../../data/articles.parquet')
        self.document_vectors_df = pl.read_parquet('../../data/document_vector.parquet')
        self.train_behaviors_df = pl.read_parquet('../../data/train/behaviors.parquet')
        self.test_behaviors_df = pl.read_parquet('../../data/validation/behaviors.parquet')

    def process_EBNeRD_dataset(self) -> list[pl.DataFrame]:
        articles_processed = self.process_dataframe(self.articles_df, None, ["article_id"], ["article_id"])
        document_vectors_processed = self.process_dataframe(self.document_vectors_df, None, None, None)
        behaviors_processed = self.process_train_test_df(self.train_behaviors_df, self.test_behaviors_df, ["impression_id", "article_id", "impression_time", "user_id"], ["article_id"], "impression_time")
        return articles_processed, document_vectors_processed, behaviors_processed

    def process_train_test_df(
            self, 
            train_df: pl.DataFrame, 
            test_df: pl.DataFrame, 
            remove_columns: list[str], 
            filter_null_columns: list[str], 
            sort_by: str
        ) -> pl.DataFrame:
        """
        Process training and testing behavior data by selecting relevant columns,
        filtering out rows with null values, and sorting by "impression_time" in descending order.

        Parameters
        ----------
        train_df : pl.DataFrame
            Training behavior DataFrame.
        test_df : pl.DataFrame
            Testing behavior DataFrame.
        remove_columns : list
            List of columns to remove.
        filter_null_columns : list
            List of columns to filter for null_value).
        sort_by : str
            List of columns to sort by.

        Returns
        -------
        pl.DataFrame
            A combined DataFrame containing processed data, optionally sorted by the sort_by column in descending order.
        """
        # Keep only relevant columns.
        processed_train_df = self.process_dataframe(train_df, remove_columns, filter_null_columns, sort_by)
        processed_test_df = self.process_dataframe(test_df, remove_columns, filter_null_columns, sort_by)
        
        # Concatenate the processed training and testing data.
        combined_df = pl.concat([processed_train_df, processed_test_df])
        return combined_df

    def process_dataframe(
        self,
        df: pl.DataFrame, 
        remove_columns: list = None, 
        filter_null_columns: list = None, 
        sort_by: str = None
    ) -> pl.DataFrame:
        """
        Processes a DataFrame with optional selection, null filtering, and sorting.

        Parameters:
        ----------
        df : (pl.DataFrame) 
            The input DataFrame.
        remove_columns : (list, optional)
            Columns to remove. If None, all columns are kept.
        filter_null_columns : (list, optional) 
            Columns to check for null values. If None, no filtering is applied.
        sort_by : (str, optional)
            Column name to sort by. If None, no sorting is applied.

        Returns:
        ----------
        pl.DataFrame 
            The processed DataFrame.
        """
        
        if remove_columns is not None:
            df = df.drop(remove_columns)

        if filter_null_columns is not None:
            df = df.filter(pl.col(filter_null_columns).is_not_null())


        if sort_by is not None:
            df = df.sort(sort_by, descending=True)

        return df

    def random_split(
            self,
            df: pl.DataFrame, 
            test_ratio: float = 0.30
        ) -> (pl.DataFrame, pl.DataFrame):
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


    def time_based_split(
            self,
            df: pl.DataFrame, 
            time_field: str = "impression_time", 
            id_field: str = "impression_id", 
            test_ratio: float = 0.30
        ) -> (pl.DataFrame, pl.DataFrame):
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
        df_sorted = df.sort([time_field, id_field])
        n_total = df_sorted.height
        # Calculate the number of rows to allocate to the test set.
        n_test = int(n_total * test_ratio)
        # Select the oldest interactions for the test set.
        test_df = df_sorted.head(n_test)
        # Use the remaining rows for the training set.
        train_df = df_sorted.tail(n_total - n_test)
        return train_df, test_df
