import pandas as pd

# Load the CSV files
audio_features_mapped = pd.read_csv(r'C:\mycode\agepredictor\datasets\audio_features_mapped.csv')
processed_audio_features_cleaned = pd.read_csv(r"C:\mycode\agepredictor\datasets\processed_audio_features.csv")
y_test_vs_y_pred = pd.read_csv(r'C:\mycode\agepredictor\datasets\y_test_vs_y_pred.csv')

# List of common feature columns to merge on
common_columns = ['chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 
                  'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 
                  'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 
                  'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20', 'Sex']

# Merge processed_audio_features_cleaned with audio_features_mapped to get the Age_Range
merged_audio = processed_audio_features_cleaned.merge(audio_features_mapped[common_columns + ['Age_Range']], 
                                                      on=common_columns, 
                                                      how='left')

# Check if the merge worked and Age_Range is correctly added
print("Merged DataFrame with Age_Range:")
print(merged_audio.head())

# Now, replace the prediction columns in y_test_vs_y_pred with Age_Range
prediction_columns = ['XGBoost_y_pred', 'Gradient Boosting_y_pred', 'Decision Tree_y_pred', 
                      'Random Forest_y_pred', 'Linear Gradient Boosting_y_pred', 'Logistic Regression_y_pred']

# Replace each prediction column with Age_Range
for col in prediction_columns:
    y_test_vs_y_pred[col] = merged_audio['Age_Range']

# Save the updated DataFrame
y_test_vs_y_pred.to_csv(r'C:\mycode\agepredictor\datasets\decoded_y_test_vs_y_pred.csv', index=False)

print("Prediction columns have been replaced with Age_Range. Output saved to 'decoded_y_test_vs_y_pred.csv'.")
