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
        misclassification_count[col] = y_test_col[misclassified].value_counts()
    
    return misclassification_count

# Get the misclassification counts
misclassified_results = check_misclassified(df)

# Display the misclassification results
for model, misclassifications in misclassified_results.items():
    print(f'{model}:')
    print(misclassifications)
    print()