"""
This file contains utility classes and functions for logging to stdout and stderr
Adapted from robomimic: https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/log_utils.py
"""

import inspect
import logging
import os
import time

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
        logging.INFO: colored(FORMAT_STR["console"], "green", attrs=["bold"]) + MESSAGE_STR,
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
            time_str = str(time.time()).replace(".", "_")
            log_file_path = "/tmp/robosuite_{}_{}.log".format(time_str, os.getpid())
            fh = logging.FileHandler(log_file_path)
            print(colored("[robosuite]: Saving logs to {}".format(log_file_path), "yellow"))
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
            logger.setLevel(logging.getLevelName(console_logging_level))

    def get_logger(self):
        """_summary_

        Returns:
            DefaultLogger: The retrieved logger whose name equals self.logger_name
        """
        logger = logging.getLogger(self.logger_name)
        return logger


def format_message(level: str, message: str) -> str:
    """
    Format a message with colors based on the level and include file and line number.

    Args:
        level (str): The logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        message (str): The message to format.

    Returns:
        str: The formatted message with file and line number.
    """
    # Get the caller's file name and line number
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno

    # Level-based coloring
    level_colors = {
        "DEBUG": "blue",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red",
    }
    attrs = ["bold"]
    if level == "CRITICAL":
        attrs.append("reverse")

    color = level_colors.get(level, "white")
    formatted_message = colored(f"[{level}] {filename}:{lineno} - {message}", color, attrs=attrs)
    return formatted_message


def rs_assert(condition: bool, message: str):
    """
    Assert a condition and raise an error with a formatted message if the condition fails.

    Args:
        condition (bool): The condition to check.
        message (str): The error message to display if the assertion fails.
    """
    if not condition:
        formatted_message = format_message("ERROR", message)
        raise AssertionError(formatted_message)


ROBOSUITE_DEFAULT_LOGGER = DefaultLogger(
    console_logging_level=macros.CONSOLE_LOGGING_LEVEL,
    file_logging_level=macros.FILE_LOGGING_LEVEL,
).get_logger()
