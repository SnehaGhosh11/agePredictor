# session_states.py

import streamlit as st 
from utils.logging_utils import *

ss = st.session_state

def initialize_session_states():
    try:
        log_message("info", "Initializing the session states.")
        
        # Initialize states with default values if not present
        if "train_model_list" not in ss:
            ss["train_model_list"] = None
        if "is_button_pressed" not in ss:
            ss["is_button_pressed"] = False
            
    except Exception as e:
        log_message("error", f"Error initializing session states: {e}")