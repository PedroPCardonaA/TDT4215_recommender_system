import polars as pl

def perform_eda(df: pl.DataFrame, name: str = "DataFrame"):
    """
    Performs an exploratory data analysis on the given Polars DataFrame.
    
    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to analyze.
    name : str
        A descriptive name of the DataFrame for print statements.
    """
    # Prints DataFrame name and schema
    print(f"=== {name} Schema ===")
    print(df.schema)
    
    # Prints summary statistics
    print(f"\n=== {name} describe() ===")
    print(df.describe())
    
    # Prints first few rows
    print(f"\n=== {name} head() ===")
    print(df.head())
    
    # Prints null counts in each column
    print(f"\n=== {name} Null Counts ===")
    null_counts = df.null_count()
    print(dict(zip(null_counts.columns, null_counts.select(pl.all()).row(0))))