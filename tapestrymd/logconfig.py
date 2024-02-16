import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
from pythonjsonlogger import jsonlogger
import os

# Define a new log level
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

class CustomLogger(logging.Logger):
    def trace(self, message, *args, **kwargs):
        if self.isEnabledFor(TRACE_LEVEL_NUM):
            self._log(TRACE_LEVEL_NUM, message, args, **kwargs)

# Configuration function
def setup_logger(name, logging_level=TRACE_LEVEL_NUM) -> CustomLogger:
    # Assert that the retrieved logger is of the correct type
    logger = logging.getLogger(name)
    assert isinstance(logger, CustomLogger), f"Logger is not of type CustomLogger. Got {type(logger)} instead."

    log_file = f'my_logs_{name}-{os.getpid()}.log'
    
    logger.setLevel(logging_level)

    if not logger.handlers:  # Check if handlers already exist
        # Concurrent Rotating file handler
        file_handler = ConcurrentRotatingFileHandler(log_file, maxBytes=1000000, backupCount=5)
        file_formatter = jsonlogger.JsonFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream handler (console output)
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter('[%(name)s]%(levelname)s: %(message)s')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    logger.propagate = False  # Prevents log messages from being duplicated in parent loggers

    return logger
