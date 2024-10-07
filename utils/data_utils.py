# data_utils.py

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
        dict: Dictionary of LabelEncoders for categorical columns.
        StandardScaler: Fitted StandardScaler object.
    """
    # Handle missing values
    df = df.dropna()
    
    # Separate categorical and numerical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Apply Label Encoding to categorical columns
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Apply Standard Scaling to numerical columns
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df, encoders, scaler

def clean_sample_data(df, encoders, scaler):
    """
    Clean the sample data using the encoders and scaler fitted on the training data.

    Args:
        df (pd.DataFrame): The raw sample dataframe.
        encoders (dict): Dictionary of LabelEncoders fitted on the training data.
        scaler (StandardScaler): StandardScaler fitted on the training data.

    Returns:
        pd.DataFrame: The cleaned and processed sample dataframe.
    """
    # Handle missing values
    df = df.dropna()

    # Apply Label Encoding to categorical columns
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
        else:
            raise ValueError(f"Column {col} missing in sample data")

    # Apply Standard Scaling to numerical columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = scaler.transform(df[numeric_columns])

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

def predict_with_models(X, models):
    """
    Function to predict using trained models.

    Args:
        X (pd.DataFrame): Feature matrix for prediction.
        models (dict): Dictionary of trained models.

    Returns:
        dict: Dictionary containing model names and their predictions.
    """
    predictions = {}
    for model_name, model in models.items():
        preds = model.predict(X)
        predictions[model_name] = preds
    return predictions

def evaluate_predictions(y_true, y_pred):
    """
    Function to evaluate predictions.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics