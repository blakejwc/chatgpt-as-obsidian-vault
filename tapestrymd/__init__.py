import logging
from .logconfig import CustomLogger

# Tell the logging module to use your custom logger
logging.setLoggerClass(CustomLogger)