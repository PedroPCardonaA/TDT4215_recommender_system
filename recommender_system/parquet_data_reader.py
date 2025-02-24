import polars as pl

class ParquetDataReader:
    """
    A simple class to read Parquet files and return a DataFrame.
    """

    def __init__(self):
        """
        Initialize the ParquetDataReader without requiring arguments.
        """
        pass

    def read_data(self, path: str) -> pl.DataFrame:
        """
        Read the Parquet file and return the DataFrame.

        Parameters
        ----------
        path : str
            Path to the Parquet file.

        Returns
        -------
        pl.DataFrame
            The DataFrame containing the Parquet data.

        Raises
        ------
        ValueError
            If the Parquet file cannot be read.
        """
        try:
            df = pl.read_parquet(path)
            return df
        except Exception as e:
            raise ValueError(f"Failed to read the Parquet file: {e}")
