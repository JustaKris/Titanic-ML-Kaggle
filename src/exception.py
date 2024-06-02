import sys
import logging
from src.logger import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Construct a detailed error message.

    Args:
        error (Exception): The exception object.
        error_detail (sys): The sys module to extract exception details.

    Returns:
        str: A formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        """
        Initialize the CustomException with a detailed error message.

        Args:
            error_message (str): The error message.
            error_detail (sys): The sys module to extract exception details.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        logging.error(self.error_message)  # Log the error message

    def __str__(self) -> str:
        """
        Return the error message string representation.
        
        Returns:
            str: The detailed error message.
        """
        return self.error_message

# Testing
# if __name__ == "__main__":
#     try:
#         a = 1 / 0
#     except Exception as e:
#         logging.info("Divide by Zero")
#         raise CustomException(e, sys)
