import pandas as pd
import os

# Hyperparameter: Number of rows to read from each file
N_ROWS = 2000  # You can change this value

# Folder containing the datasets
folder_path = 'datasets'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Read the first N_ROWS from each CSV and collect them
df_list = [pd.read_csv(os.path.join(folder_path, file), nrows=N_ROWS) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Shuffle the combined DataFrame
shuffled_df = combined_df.sample(frac=1, random_state=8).reset_index(drop=True)

# Save the shuffled DataFrame
shuffled_df.to_csv('dataset.csv', index=False)

print(f"First {N_ROWS} rows from each dataset combined, shuffled, and saved as 'dataset.csv'")
