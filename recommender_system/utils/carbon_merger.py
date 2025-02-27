"""
This module merges CSV files from the outputdirectory and extracts selected columns. Then it
writes the combined data to a summary file. It infers `model` and `method`
from each CSV filename and includes them as columns in the final output.
"""

import os
import glob
import pandas as pd

def main():
    """
    Merge CSV files from the `/demostrations/output` folder, extract specific columns,
    and save the combined result as a single CSV file in `/output/summary/`.

    This function:
        1. Locates all CSV files in the `/demostrations/output` directory.
        2. Infers `model` and `method` from each filename (e.g., `baseline_fit_emission.csv`).
        3. Keeps only a predefined list of columns, inserting `model` and `method` first.
        4. Stacks all CSV files vertically into one DataFrame.
        5. Saves the merged DataFrame as `Carbon_Sheet_Summary.csv` in the `/output/summary` folder.

    Raises
    ------
    FileNotFoundError
        If no CSV files are found in the specified directory.
    Exception
        If any error occurs while reading or merging the DataFrames.

    Returns
    -------
    None
        The function prints the location of the merged CSV file and does not return any object.
    """
    # 1. Define which columns we want to keep (in the final order).
    #    Note that 'model' and 'method' will be added first.
    KEEP_COLUMNS = [
        "timestamp",
        "run_id",
        "duration",
        "emissions",
        "emissions_rate",
        "cpu_power",
        #"gpu_power",
        "ram_power",
        "cpu_energy",
        #"gpu_energy",
        "ram_energy",
        "energy_consumed",
        "country_name",
        "cpu_count",
        "cpu_model",
        #"gpu_count",
        #"gpu_model",
        "ram_total_size"
    ]
    
    # 2. Compute the /output folder path relative to this script.
    base_folder = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "demostrations", "output")
    )
    
    # 3. Find all CSV files directly in the /output folder.
    input_pattern = os.path.join(base_folder, "*.csv")
    csv_files = glob.glob(input_pattern)
    
    if not csv_files:
        print("No CSV files found in the specified folder.")
        return
    
    dataframes = []
    
    for file in csv_files:
        try:
            # Example filename: "item_based_fit_emission.csv"
            # Remove the trailing "_emission" if present.
            base_name = os.path.splitext(os.path.basename(file))[0]
            if base_name.endswith("_emission"):
                base_name = base_name[: -len("_emission")]  # remove "_emission"

            # Split on underscores to separate model from method
            # e.g. "item_based_fit" -> ["item", "based", "fit"]
            parts = base_name.split("_")
            
            # Last part is typically the method ("fit", "recommend", etc.)
            method = parts[-1] if len(parts) > 1 else "unknown_method"
            
            # Everything else is the model ("item_based", "baseline", etc.)
            model = "_".join(parts[:-1]) if len(parts) > 1 else "unknown_model"
            
            # 4. Read the CSV into a DataFrame.
            df = pd.read_csv(file)
            
            # 5. Reindex the DataFrame to keep only the columns we need.
            #    This creates any missing columns with NaN if they're not in the CSV.
            df = df.reindex(columns=KEEP_COLUMNS)
            
            # 6. Insert 'model' and 'method' as new columns.
            df.insert(0, "model", model)   # Put 'model' in the first column
            df.insert(1, "method", method) # Put 'method' in the second column
            
            dataframes.append(df)
        
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    if not dataframes:
        print("No valid DataFrames to merge.")
        return
    
    # 7. Concatenate all DataFrames (stack them vertically).
    merged_df = pd.concat(dataframes, axis=0, ignore_index=True)
    
    # 8. Create the /summary folder if it does not exist.
    output_dir = os.path.join(base_folder, "summary")
    os.makedirs(output_dir, exist_ok=True)
    
    # 9. Save the final merged DataFrame to a CSV file.
    output_file = os.path.join(output_dir, "Carbon_Sheet_Summary.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV file created at {output_file}")

if __name__ == "__main__":
    main()
