import logging
import os
import sys

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Function to setup a logger; can be used across different modules.
    
    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file. If None, logs will only go to console.
        level (int): Logging level.
        
    Returns:
        logging.Logger: Configured logger.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler
        if log_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger
