import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = r"C:\mycode\agepredictor\datasets\processed_audio_features.csv"
df = pd.read_csv(file_path)

# Assuming 'Age_Range' is the target column and the rest are features
X = df.drop(columns=['Age_Range'])  # Features
y = df['Age_Range']  # Target

# Split the dataset using stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Save the train and test sets to CSV files
X_train.to_csv('C:/mycode/agepredictor/datasets/X_train.csv', index=False)
X_test.to_csv('C:/mycode/agepredictor/datasets/X_test.csv', index=False)
y_train.to_csv('C:/mycode/agepredictor/datasets/y_train.csv', index=False, header=True)
y_test.to_csv('C:/mycode/agepredictor/datasets/y_test.csv', index=False, header=True)

print("Train and test sets saved to CSV files.")

# Define models
models = {
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Linear Gradient Boosting': HistGradientBoostingClassifier()  # Linear Gradient Boosting
}

# Train and evaluate models
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")