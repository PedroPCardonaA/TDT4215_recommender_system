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


import polars as pl

def data_sparsity(behavior_df: pl.DataFrame):
    """
    Calculates the sparsity of a user-item interaction DataFrame.
    Duplicate interactions (same user_id and article_id) are removed.

    Parameters
    ----------
    behavior_df : pl.DataFrame
        The user-item interaction DataFrame.

    Returns
    -------
    float
        The sparsity of the DataFrame.
    """
    unique_behavior_df = behavior_df.unique(subset=["user_id", "article_id"])
    num_interactions = unique_behavior_df.shape[0]
    num_users = unique_behavior_df["user_id"].n_unique()
    num_items = unique_behavior_df["article_id"].n_unique()
    sparsity = 1 - (num_interactions / (num_users * num_items))
    return sparsity
