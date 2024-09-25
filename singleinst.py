import pandas as pd

# Load the dataset from the given path
file_path = 'C:/mycode/agepredictor/datasets/processed_audio_features_cleaned.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Check the unique values in the Age_Range column
unique_classes = df['Age_Range'].unique()
print(f"Unique Age_Range classes: {unique_classes}")

# Filter out the single instance classes based on Age_Range
single_instance_classes = df.groupby('Age_Range').filter(lambda x: len(x) == 1)

# Save the filtered single instance class to a new CSV file
output_path = 'C:/mycode/agepredictor/datasets/single_instance_classes.csv'
single_instance_classes.to_csv(output_path, index=False)

print(f"Single instance classes saved to {output_path}")