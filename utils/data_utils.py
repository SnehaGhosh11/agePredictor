# data_utils.py

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    """
    Load data from a given file path.
    Assumes CSV format for simplicity.
    
    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        raise IOError(f"Error loading file: {e}")

def clean_data(df):
    """
    Function to clean the dataframe.
    This function handles missing values, encodes categorical features,
    and scales numeric features using Standard Scaler.

    Args:
        df (pd.DataFrame): The raw dataframe.

    Returns:
        pd.DataFrame: The cleaned and processed dataframe.
    """
    # Handle missing values
    df.dropna(inplace=True)
    
    # Separate categorical and numerical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Apply Label Encoding to categorical columns
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Apply Standard Scaling to numerical columns
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df

def train_models(X, y, models):
    """
    Function to train models with the provided data.
    Returns a dictionary with model names and their accuracy scores.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        models (dict): Dictionary of models to train.

    Returns:
        dict: Dictionary containing model names and their accuracy scores.
    """
    accuracy_results = {}
    for model_name, model in models.items():
        model.fit(X, y)
        accuracy = model.score(X, y)  # Calculate accuracy on training data
        accuracy_results[model_name] = accuracy
    return accuracy_results