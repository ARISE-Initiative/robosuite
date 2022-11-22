"""
This file contains utility classes and functions for logging to stdout and stderr
Adapted from robomimic: https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/log_utils.py
"""
import logging

from termcolor import colored

import robosuite.macros as macros


class RobosuiteColorFormatter(logging.Formatter):
    format_str = "[robosuite %(levelname)s] "
    message_str = "%(message)s (%(filename)s:%(lineno)d)"
    FORMATS = {
        logging.DEBUG: format_str + message_str,
        logging.INFO: "%(message)s",
        logging.WARNING: colored(format_str, "yellow", attrs=["bold"]) + message_str,
        logging.ERROR: colored(format_str, "red", attrs=["bold"]) + message_str,
        logging.CRITICAL: colored(format_str, "red", attrs=["bold", "reverse"]) + message_str,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class RobosuiteFileFormatter(logging.Formatter):
    format_str = "[robosuite %(levelname)s - %(asctime)s] "
    message_str = "%(message)s (%(filename)s:%(lineno)d)"
    FORMATS = {
        logging.DEBUG: format_str + message_str,
        logging.INFO: format_str + message_str,
        logging.WARNING: colored(format_str, "yellow", attrs=["bold"]) + message_str,
        logging.ERROR: colored(format_str, "red", attrs=["bold"]) + message_str,
        logging.CRITICAL: colored(format_str, "red", attrs=["bold", "reverse"]) + message_str,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class RobosuiteConsoleFormatter(logging.Formatter):
    format_str = "[robosuite %(levelname)s] "
    message_str = "%(message)s (%(filename)s:%(lineno)d)"
    FORMATS = {
        logging.DEBUG: format_str + message_str,
        logging.INFO: "%(message)s",
        logging.WARNING: colored(format_str, "yellow", attrs=["bold"]) + message_str,
        logging.ERROR: colored(format_str, "red", attrs=["bold"]) + message_str,
        logging.CRITICAL: colored(format_str, "red", attrs=["bold", "reverse"]) + message_str,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class RobosuiteDefaultLogger:
    def __init__(self, logger_name="robosuite_logs", console_logging_level="INFO", file_logging_level=None):
        self.logger_name = logger_name
        logger = logging.getLogger(self.logger_name)

        if file_logging_level is not None:
            fh = logging.FileHandler("/tmp/robosuite.log")
            fh.setLevel(logging.getLevelName(file_logging_level))
            file_formatter = RobosuiteFileFormatter()
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)

        if console_logging_level is not None:
            ch = logging.StreamHandler()
            ch.setLevel(logging.getLevelName(console_logging_level))
            console_formatter = RobosuiteConsoleFormatter()
            ch.setFormatter(console_formatter)
            logger.addHandler(ch)

    def get_logger(self):
        logger = logging.getLogger(self.logger_name)
        return logger


ROBOSUITE_DEFAULT_LOGGER = RobosuiteDefaultLogger(
    console_logging_level=macros.CONSOLE_LOGGING_LEVEL,
    file_logging_level=macros.FILE_LOGGING_LEVEL,
).get_logger()
