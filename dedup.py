import pandas as pd

# Define the file path
file_path = r"C:\mycode\agepredictor\datasets\audio_features_mapped.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Function to classify columns as Numerical or Categorical
def classify_columns(df):
    column_classification = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            column_classification[column] = 'Numerical'
        else:
            column_classification[column] = 'Categorical'
    return column_classification

# Function to remove duplicate columns
def remove_duplicate_columns(df):
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# Remove duplicate columns
df = remove_duplicate_columns(df)

# Classify the columns
column_types = classify_columns(df)

# Display the classification of columns and cleaned dataframe
print("Column Classification:")
for column, col_type in column_types.items():
    print(f"{column}: {col_type}")

# Optional: If you'd like to see the first few rows of the cleaned dataframe
print("\nFirst 5 rows of the deduplicated dataframe:")
print(df.head())