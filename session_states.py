import streamlit as st
from utils.logging_utils import *

ss = st.session_state

def initialize_session_states():
    try:
        log_message("info", f"Initializing the states.")
        if "train_model_list" not in ss:
            ss["train_model_list"] = None
            
    except Exception as e:
        log_message("info", f"session_state.json not found. Initializing session state.{e}")