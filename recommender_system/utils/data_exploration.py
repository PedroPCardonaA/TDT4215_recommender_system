from IPython.display import display
import polars as pl

def perform_eda(df: pl.DataFrame, name: str = "DataFrame"):
    """
    Performs an exploratory data analysis on the given Polars DataFrame,
    displaying the results in a Jupyter-friendly format.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to analyze.
    name : str
        A descriptive name of the DataFrame for print statements.
    """
    # Displays DataFrame name
    print(f"=== {name} ===\n")

    # 1. Display schema (converted to a small Polars -> Pandas table)
    print("-- Schema --")
    schema_data = {
        "Column": list(df.schema.keys()),
        "Dtype": [str(dtype) for dtype in df.schema.values()]
    }
    schema_df = pl.DataFrame(schema_data)
    display(schema_df.to_pandas())

    # 2. Display summary statistics as a table
    print("\n-- Describe --")
    describe_df = df.describe()
    display(describe_df.to_pandas())

    # 3. Display the first few rows
    print("\n-- Head --")
    head_df = df.head()
    display(head_df.to_pandas())

    # 4. Display null counts in each column
    print("\n-- Null Counts --")
    null_counts_df = df.null_count()
    display(null_counts_df.to_pandas())