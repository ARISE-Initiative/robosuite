"""
This file contains utility classes and functions for logging to stdout and stderr
Adapted from robomimic: https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/log_utils.py
"""
import logging

from termcolor import colored

import robosuite.macros as macros

LEVEL_COLORS = {
    logging.DEBUG: "green",
    logging.INFO: "green",
    logging.WARNING: "yellow",
    logging.ERROR: "red",
    logging.CRITICAL: "red",
}

FORMAT_STR = {"file": "[robosuite %(levelname)s - %(asctime)s] ", "console": "[robosuite %(levelname)s] "}

MESSAGE_STR = "%(message)s (%(filename)s:%(lineno)d)"


class FileFormatter(logging.Formatter):
    """Formatter class of logging for file logging."""

    FORMATS = {
        levelno: colored(FORMAT_STR["file"], color, attrs=["bold"]) + MESSAGE_STR
        for (levelno, color) in LEVEL_COLORS.items()
    }

    def format(self, record):
        """Apply custom fomatting on LogRecord object record."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class ConsoleFormatter(logging.Formatter):
    """Formatter class of logging for console logging."""

    FORMATS = {
        logging.DEBUG: FORMAT_STR["console"] + MESSAGE_STR,
        logging.INFO: "%(message)s",
        logging.WARNING: colored(FORMAT_STR["console"], "yellow", attrs=["bold"]) + MESSAGE_STR,
        logging.ERROR: colored(FORMAT_STR["console"], "red", attrs=["bold"]) + MESSAGE_STR,
        logging.CRITICAL: colored(FORMAT_STR["console"], "red", attrs=["bold", "reverse"]) + MESSAGE_STR,
    }

    def format(self, record):
        """Apply custom fomatting on LogRecord object record."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class DefaultLogger:
    """Default logger class in robosuite codebase."""

    def __init__(self, logger_name="robosuite_logs", console_logging_level="INFO", file_logging_level=None):
        """
        Args:
            logger_name (str, optional): logger name. Defaults to "robosuite_logs".
            console_logging_level (str, optional): logging level for console logging. Defaults to "INFO".
            file_logging_level (_type_, optional): logging level for file logging. Defaults to None.
        """
        self.logger_name = logger_name
        logger = logging.getLogger(self.logger_name)

        if file_logging_level is not None:
            fh = logging.FileHandler("/tmp/robosuite.log")
            fh.setLevel(logging.getLevelName(file_logging_level))
            file_formatter = FileFormatter()
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)

        if console_logging_level is not None:
            ch = logging.StreamHandler()
            ch.setLevel(logging.getLevelName(console_logging_level))
            console_formatter = ConsoleFormatter()
            ch.setFormatter(console_formatter)
            logger.addHandler(ch)

    def get_logger(self):
        """_summary_

        Returns:
            DefaultLogger: The retrieved logger whose name equals self.logger_name
        """
        logger = logging.getLogger(self.logger_name)
        return logger


ROBOSUITE_DEFAULT_LOGGER = DefaultLogger(
    console_logging_level=macros.CONSOLE_LOGGING_LEVEL,
    file_logging_level=macros.FILE_LOGGING_LEVEL,
).get_logger()
