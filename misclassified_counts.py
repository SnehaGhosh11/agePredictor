import pandas as pd

# Load the CSV file from the given path
file_path = r'C:\mycode\agepredictor\datasets\modified_y_test_vs_pred.csv'
df = pd.read_csv(file_path)

# Function to check misclassified class for each column
def check_misclassified(df):
    y_test_col = df['y_test']
    misclassification_count = {}
    
    # Compare y_test with other columns
    for col in df.columns[1:]:
        misclassified = (y_test_col != df[col])
        misclassification_count[col] = y_test_col[misclassified].value_counts().sum()  # Sum the misclassifications
    
    return misclassification_count

# Get the misclassification counts
misclassified_results = check_misclassified(df)

# Prepare data for CSV output
misclassified_data = []

# Convert the misclassification results into a list of rows for CSV
for model, total_misclassifications in misclassified_results.items():
    misclassified_data.append([model, total_misclassifications])

# Convert to a DataFrame for saving
misclassified_df = pd.DataFrame(misclassified_data, columns=['Model', 'Total Misclassifications'])

# Save to CSV
output_path = r'C:\mycode\agepredictor\datasets\misclassified_counts.csv'
misclassified_df.to_csv(output_path, index=False)

# Find the model with the least misclassifications
least_misclassified_model = misclassified_df.loc[misclassified_df['Total Misclassifications'].idxmin()]

# Print the model with the least misclassification
print(f"Model with least misclassifications: {least_misclassified_model['Model']} with {least_misclassified_model['Total Misclassifications']} misclassifications")

print(f"Misclassification results saved to {output_path}")