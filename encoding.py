import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
file_path = r"C:\mycode\agepredictor\datasets\audio_features_mapped.csv"
df = pd.read_csv(file_path)

# Drop columns with "Unnamed" if they are empty or irrelevant
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Handle missing values (NaNs) by filling them with the mean for numerical or the mode for categorical columns
for column in df.columns:
    if df[column].isnull().any():
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column].fillna(df[column].mean(), inplace=True)  # Fill NaNs with the column mean for numeric data
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)  # Fill NaNs with the mode for categorical data

# Initialize LabelEncoder and StandardScaler
label_encoders = {}
scaler = StandardScaler()

# Loop through the columns and apply transformations
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        # Apply Standard Scaling for numeric columns
        df[column] = scaler.fit_transform(df[[column]].values)
    else:
        # Apply Label Encoding for categorical columns
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

# Show the transformed dataframe
print("\nTransformed DataFrame:")
print(df.head())

# Save the processed dataframe
processed_file_path = r"C:\mycode\agepredictor\datasets\processed_audio_features.csv"
df.to_csv(processed_file_path, index=False)
print(f"Processed dataset saved to: {processed_file_path}")
