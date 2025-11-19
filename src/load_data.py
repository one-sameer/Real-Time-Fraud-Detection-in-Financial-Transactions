# This module handles loading the dataset into a pandas DataFrame.

import pandas as pd
import os
try:    
    from src.config import DATA_FILE
except ModuleNotFoundError:
    from config import DATA_FILE

def load_dataset(filepath: str = DATA_FILE) -> pd.DataFrame:

    # If the path is invalid
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    
    # Loaading the dataset into DataFrame
    df = pd.read_csv(filepath)
    return df

# Main function for verification
if __name__ == "__main__":
    data = load_dataset()
    print(data.head())