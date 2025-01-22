import pandas as pd

class ParquetDataReader:
    """
    A simple class to read Parquet files and return a DataFrame.
    """

    def __init__(self):
        pass



    def read_data(self, path: str) -> pd.DataFrame:
        """
        Read the Parquet file and return the DataFrame.

        Returns:
            pd.DataFrame: The DataFrame containing the Parquet data.
        """
        try:
            df = pd.read_parquet(path)
            return df
        except Exception as e:
            raise ValueError(f"Failed to read the Parquet file: {e}")