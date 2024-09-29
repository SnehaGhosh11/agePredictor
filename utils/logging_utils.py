import logging
import os
import inspect
from consts.file_consts import *

""" 
Here's how the log levels work:

logging.DEBUG: Logs all messages (DEBUG, INFO, WARNING, ERROR, CRITICAL).
logging.INFO: Logs INFO, WARNING, ERROR, and CRITICAL messages, but not DEBUG.
logging.WARNING: Logs WARNING, ERROR, and CRITICAL messages, but not DEBUG or INFO.
logging.ERROR: Logs ERROR and CRITICAL messages, but not DEBUG, INFO, or WARNING.
logging.CRITICAL: Logs only CRITICAL messages. 

"""
# Configure logging
logging.basicConfig(filename='app.log', 
                    filemode='w', 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %I:%M:%S %p')

def log_message(level, message):
    try:
        # Get the current stack frame
        stack = inspect.stack()
        # The caller of log_message is one level up the stack from the current position
        caller_frame = stack[1]
        caller_filename = os.path.splitext(os.path.basename(caller_frame.filename))[0]
        logger = logging.getLogger(caller_filename)
        logger.setLevel(logging.DEBUG)
    
        full_message = f'{level} message from {caller_filename}: {message}'
        
        log_function = getattr(logger, level)
        log_function(full_message)
    except Exception as e:
        print(f"An error occurred while logging: {e}")