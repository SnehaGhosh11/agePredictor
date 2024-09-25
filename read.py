import pandas as pd

# Define the file path
file_path = r"C:\mycode\agepredictor\datasets\audio_features_mapped.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(df.head())