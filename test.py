import pandas as pd

# Load the CSV file
df = pd.read_csv("DatasetFinal.csv")

# Add the new column
df["design_name"] = "bc0"

# Keep only the first 15,000 rows
df = df.head(3000)

# Save the updated DataFrame to a new CSV
df.to_csv("dataset.csv", index=False)
