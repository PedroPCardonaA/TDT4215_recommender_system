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
        self.document_vectors_df = pl.read_parquet(
            '../../data/document_vector.parquet')
        self.train_behaviors_df = pl.read_parquet(
            '../../data/train/behaviors.parquet')
        self.test_behaviors_df = pl.read_parquet(
            '../../data/validation/behaviors.parquet')

    def baseline_process_EBNeRD(self) -> list[pl.DataFrame]:
        """
        Preprocesses the EBNeRD dataset by cleaning and transforming articles, document vectors, and behaviors data.

        Returns
        -------
        list[pl.DataFrame]
            A list containing three processed DataFrames:
            1. Processed articles DataFrame with selected columns removed and sorted by published time.
            2. Document vectors DataFrame (returned as is, without preprocessing).
            3. Processed behaviors DataFrame with missing values handled, unnecessary columns removed, 
            and missing scroll percentages imputed using the mean.
        """
        # Article Dataframe
        # Here we only need to remove non-needed values, and sort by published_time
        article_drop = [
            "total_inviews", "total_pageviews", "total_read_time", "image_ids"
        ]
        articles_processed = self.process_dataframe(
            df=self.articles_df,
            remove_columns=article_drop,
            sort_by="published_time")

        # Document vectors
        # This dataframe does not need any preprocessing
        document_vectors_processed = self.document_vectors_df

        # Behaviours DataFrame
        # We filter out columns with too much missing values and columns with non-needed future data
        # We also remove rows where article_id is null as this is the main page
        # Lastly we predict missing scroll_percentage with the mean value
        behaviour_drop = [
            "gender", "postcode", "age", "next_read_time",
            "next_scroll_percentage", "article_ids_inview",
            "article_ids_clicked"
        ]
        behaviour_non_null = ["article_id"]
        behaviour_predict = ["scroll_percentage"]

        behaviors_processed = self.process_train_test_df(
            train_df=self.train_behaviors_df,
            test_df=self.test_behaviors_df,
            remove_columns=behaviour_drop,
            filter_null_columns=behaviour_non_null,
            predict_columns=behaviour_predict)

        return articles_processed, document_vectors_processed, behaviors_processed

    def collaborative_filtering_preprocess(self):
        behaviour_drop = [
            "gender", "postcode", "age", "next_read_time",
            "next_scroll_percentage", "article_ids_inview",
            "article_ids_clicked"
        ]
        behaviour_non_null = ["article_id"]
        behaviour_predict = ["scroll_percentage"]

        behaviors_processed = self.process_train_test_df(
            train_df=self.train_behaviors_df,
            test_df=self.test_behaviors_df,
            remove_columns=behaviour_drop,
            filter_null_columns=behaviour_non_null,
            predict_columns=behaviour_predict)

        return articles_processed, document_vectors_processed, behaviors_processed

    def process_train_test_df(self,
                              train_df: pl.DataFrame,
                              test_df: pl.DataFrame,
                              remove_columns: list[str] = None,
                              filter_null_columns: list[str] = None,
                              expand_columns: list[str] = None,
                              predict_columns: list[str] = None,
                              predict_strat: str = "mean",
                              sort_by: str = None) -> pl.DataFrame:
        """
        Processes training and testing behavior data by applying column selection, 
        null filtering, optional expansion, and sorting. The processed data from 
        both training and testing sets are concatenated into a single DataFrame.

        Parameters
        ----------
        train_df : pl.DataFrame
            The training behavior DataFrame.
        test_df : pl.DataFrame
            The testing behavior DataFrame.
        remove_columns : list[str], optional
            List of column names to remove from the DataFrame. If None, no columns are removed.
        filter_null_columns : list[str], optional
            List of columns to filter by removing rows containing null values. If None, no filtering is applied.
        expand_columns : list[str], optional
            List of columns containing string values to expand into multiple boolean columns. If None, no expansion is applied.
        predict_columns : list[str], optional
            List of columns for which prediction strategies are applied. If None, no predictions are performed.
        predict_strat : str, optional
            The prediction strategy to use for missing values in predict_columns. Defaults to "mean".
        sort_by : str, optional
            Column name to sort by in descending order. If None, no sorting is applied.

        Returns
        -------
        pl.DataFrame
            A concatenated DataFrame containing processed data from both training and testing sets.
        """
        processed_train_df = self.process_dataframe(train_df, remove_columns,
                                                    filter_null_columns,
                                                    expand_columns,
                                                    predict_columns,
                                                    predict_strat, sort_by)
        processed_test_df = self.process_dataframe(test_df, remove_columns,
                                                   filter_null_columns,
                                                   expand_columns,
                                                   predict_columns,
                                                   predict_strat, sort_by)

        return pl.concat([processed_train_df, processed_test_df])

    def process_dataframe(self,
                          df: pl.DataFrame,
                          remove_columns: list[str] = None,
                          filter_null_columns: list[str] = None,
                          expand_columns: list[str] = None,
                          predict_columns: list[str] = None,
                          predict_strat: str = "mean",
                          sort_by: str = None) -> pl.DataFrame:
        """
        Processes a DataFrame by applying optional column removal, null filtering, 
        column expansion, prediction strategy application, and sorting.

        Parameters
        ----------
        df : pl.DataFrame
            The input DataFrame to be processed.
        remove_columns : list[str], optional
            List of columns to remove. If None, all columns are retained.
        filter_null_columns : list[str], optional
            List of columns to check for null values. Rows with null values in these columns are removed. If None, no filtering is applied.
        expand_columns : list[str], optional
            List of columns containing string values to be expanded into multiple boolean columns. If None, no expansion is applied.
        predict_columns : list[str], optional
            List of columns where missing values will be filled based on a prediction strategy. If None, no imputation is applied.
        predict_strat : str, optional
            The strategy for handling missing values in predict_columns. Defaults to "mean".
        sort_by : str, optional
            Column name to sort by in descending order. If None, no sorting is applied.

        Returns
        -------
        pl.DataFrame
            The processed DataFrame after applying the specified transformations.
        """
        if remove_columns is not None:
            df = df.drop(remove_columns)

        if filter_null_columns is not None:
            df = df.filter(pl.col(filter_null_columns).is_not_null())

        if sort_by is not None:
            df = df.sort(sort_by, descending=True)

        return df

    def random_split(self,
                     df: pl.DataFrame,
                     test_ratio: float = 0.30) -> (pl.DataFrame, pl.DataFrame):
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
            test_ratio: float = 0.30) -> (pl.DataFrame, pl.DataFrame):
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
