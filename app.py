# app.py

import streamlit as st
from consts.model_consts import CLASSIFIERS
from utils.file_utils import save_input_file, initialize_directories
from utils.data_utils import (
    load_data,
    clean_data,
    train_models,
    predict_with_models,
    evaluate_predictions,
    clean_sample_data
)
from session_states import *
from utils.logging_utils import log_message
import pandas as pd

# Constants (ensure these are defined or imported from your project structure)
INPUT_TRAIN_FILE = "train.csv"  # Define the path to save the uploaded training file
INPUT_SAMPLE_FILE = "sample.csv"  # Define the path to save the uploaded sample file

st.title("Model Trainer")

# Initialize directories and session states
try:
    initialize_directories()
    initialize_session_states()
except OSError as e:
    log_message("error", e)

# File uploader for training data
uploaded_train_file = st.file_uploader("Choose the Training file", type=["csv"])

# Model selection
model_names = [model_name for model_name in CLASSIFIERS.keys()]
ss["train_model_list"] = st.multiselect("Select models to train data on:", model_names)

if st.button("Submit"):
    ss["is_button_pressed"] = True  # Set button pressed to true on submit

    if uploaded_train_file is None or ss["train_model_list"] == []:
        st.error("Please select at least one training file and at least one model to train on!")
        st.stop()

    try:
        # Save uploaded file
        if save_input_file(uploaded_train_file, INPUT_TRAIN_FILE):
            st.success(f"File '{uploaded_train_file.name}' uploaded and saved successfully!")

            try:
                # Load the data from the file
                df = load_data(INPUT_TRAIN_FILE)
                st.write("Data loaded successfully!")
                st.dataframe(df.head())  # Display the first few rows of the data
                
                # Clean the data (includes Label Encoding and Standard Scaling)
                df_cleaned, encoders, scaler = clean_data(df)
                st.write("Data cleaned and processed successfully!")
                st.dataframe(df_cleaned.head())  # Display the cleaned data

                # Automatically select the last column as the target (assuming it's the target column)
                target_column = df_cleaned.columns[-1]
                X = df_cleaned.drop(target_column, axis=1)
                y = df_cleaned[target_column]
                
                st.write("Data prepared for training!")
                st.write(f"Features shape: {X.shape}, Target shape: {y.shape}")

                # Get selected models
                selected_models = {model_name: CLASSIFIERS[model_name] for model_name in ss["train_model_list"]}

                # Train models and display accuracy table
                accuracy_results = train_models(X, y, selected_models)
                accuracy_df = pd.DataFrame(list(accuracy_results.items()), columns=["Model", "Accuracy"])
                
                st.write("Model Training Completed!")
                st.table(accuracy_df)  # Display accuracy table

                # Filter models with accuracy above 0.95
                high_accuracy_models = {
                    model_name: selected_models[model_name]
                    for model_name, accuracy in accuracy_results.items()
                    if accuracy >= 0.95
                }
                if high_accuracy_models:
                    st.write("Models with accuracy above 0.95:")
                    st.write(list(high_accuracy_models.keys()))

                    # File uploader for sample data
                    uploaded_sample_file = st.file_uploader(
                        "Choose the Sample dataset file for testing", type=["csv"], key="sample_file_uploader"
                    )

                    if uploaded_sample_file is not None:
                        # Save uploaded sample file
                        if save_input_file(uploaded_sample_file, INPUT_SAMPLE_FILE):
                            st.success(f"Sample file '{uploaded_sample_file.name}' uploaded and saved successfully!")

                            try:
                                # Load the sample data
                                df_sample = load_data(INPUT_SAMPLE_FILE)
                                st.write("Sample data loaded successfully!")
                                st.dataframe(df_sample.head())  # Display the first few rows of the sample data

                                # Clean the sample data (same process as training data)
                                df_sample_cleaned = clean_sample_data(df_sample, encoders, scaler)
                                st.write("Sample data cleaned and processed successfully!")
                                st.dataframe(df_sample_cleaned.head())  # Display the cleaned sample data

                                # If target column exists in sample data, separate it
                                if target_column in df_sample_cleaned.columns:
                                    X_sample = df_sample_cleaned.drop(target_column, axis=1)
                                    y_sample = df_sample_cleaned[target_column]
                                    has_target = True
                                else:
                                    X_sample = df_sample_cleaned
                                    y_sample = None
                                    has_target = False

                                st.write("Sample data prepared for prediction!")
                                st.write(f"Sample features shape: {X_sample.shape}")

                                # Predict using the high accuracy models
                                predictions = predict_with_models(X_sample, high_accuracy_models)

                                # Display predictions and evaluation metrics
                                for model_name, preds in predictions.items():
                                    st.write(f"Predictions using {model_name}:")
                                    st.write(preds)

                                    # If actual target values are available, evaluate predictions
                                    if has_target:
                                        metrics = evaluate_predictions(y_sample, preds)
                                        st.write(f"Evaluation metrics for {model_name}:")
                                        st.write(metrics)
                            except Exception as e:
                                st.error(f"Error processing sample data: {e}")
                                log_message("error", str(e))
                        else:
                            st.error("Sample file could not be saved for processing")
                    else:
                        st.info("Please upload a sample dataset CSV file to test the high accuracy models.")
                else:
                    st.info("No models with accuracy above 0.95 were found.")

            except Exception as e:
                st.error(f"Error processing data: {e}")
                log_message("error", str(e))

        else:
            st.error("File could not be saved for processing")
            st.stop()

    except Exception as e:
        log_message("error", str(e))
        st.error(f"Error uploading the file: {e}")
