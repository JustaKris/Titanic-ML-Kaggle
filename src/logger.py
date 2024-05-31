import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Define log file name using current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define log directory and ensure it exists
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Define full log file path
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE_PATH, maxBytes=10**6, backupCount=10),
        logging.StreamHandler()
    ]
)

# Testing
# if __name__ == "__main__":
#     logging.info("Logging has started")
