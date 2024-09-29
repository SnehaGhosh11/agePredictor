import streamlit as st
from consts.model_consts import CLASSIFIERS
from utils.file_utils import save_input_file, initialize_directories
from utils.data_utils import load_data, clean_data, train_models
from session_states import *
from utils.logging_utils import log_message
import pandas as pd  # Ensure pandas is imported

# Constants (ensure these are defined or imported from your project structure)
INPUT_TRAIN_FILE = "train.csv"  # Define the path to save the uploaded training file

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
                df_cleaned = clean_data(df)
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

            except Exception as e:
                st.error(f"Error processing data: {e}")
                log_message("error", str(e))

        else:
            st.error("File could not be saved for processing")
            st.stop()

    except Exception as e:
        log_message("error", str(e))
        st.error(f"Error uploading the file: {e}")