import pandas as pd
import os
import shutil
from consts.file_consts import *
from utils.file_utils import *
from utils.logging_utils import *

def initialize_directories():
    """
    Ensures that all necessary directories specified in constants are present.
    If a directory does not exist, it will be created.
    If a directory exists, it will be cleared of its contents.

    Directories are imported from the consts module.
    """
    # List all directory constants imported from consts
    directories = [INPUT_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER, MODELS_FOLDER]  # Add any new constants here

    for directory_path in directories:
        if os.path.exists(directory_path):
            # Clear the directory if it exists
            shutil.rmtree(directory_path)
            log_message("info", f"Directory '{directory_path}' cleared successfully.")
        
        # Recreate the directory
        os.makedirs(directory_path)
        log_message("info", f"Directory '{directory_path}' created successfully.")

def save_input_file(uploaded_file, save_location):
    try:

        with open(save_location, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return 1

    except Exception as e:
        log_message("error",f"Error while saving input file: {e}")
        return 0