import pandas as pd

# Define the file path
file_path = r"C:\mycode\agepredictor\datasets\audio_features_mapped.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Function to classify column types
def classify_columns(df):
    column_classification = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            column_classification[column] = 'Numerical'
        else:
            column_classification[column] = 'Categorical'
    return column_classification

# Classify the columns
column_types = classify_columns(df)

# Display the classification of columns
print(column_types)