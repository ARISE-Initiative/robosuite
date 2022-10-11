"""
This file contains utility classes and functions for logging to stdout and stderr
Adapted from robomimic: https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/log_utils.py
"""
import textwrap
from termcolor import colored

# global list of warning messages can be populated with @log_warning and flushed with @flush_warnings
WARNINGS_BUFFER = []


def log_warning(message, color="yellow", print_now=True):
    """
    This function logs a warning message by recording it in a global warning buffer.
    The global registry will be maintained until @flush_warnings is called, at
    which point the warnings will get printed to the terminal.
    Args:
        message (str): warning message to display
        color (str): color of message - defaults to "yellow"
        print_now (bool): if True (default), will print to terminal immediately, in
            addition to adding it to the global warning buffer
    """
    global WARNINGS_BUFFER
    buffer_message = colored("ROBOSUITE WARNING(\n{}\n)".format(textwrap.indent(message, "    ")), color)
    WARNINGS_BUFFER.append(buffer_message)
    if print_now:
        print(buffer_message)


def flush_warnings():
    """
    This function flushes all warnings from the global warning buffer to the terminal and
    clears the global registry.
    """
    global WARNINGS_BUFFER
    for msg in WARNINGS_BUFFER:
        print(msg)
    WARNINGS_BUFFER = []