# ==========================================================
# File: src/logger.py
# ==========================================================
"""
Production logging configuration.

Features
--------
✓ Console logging
✓ File logging
✓ Rotating log files
✓ Timestamped messages
✓ Colored log levels (console)
✓ Separate loggers for training/API
✓ No duplicate handlers
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


# ----------------------------------------------------------
# Create logs directory
# ----------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# Console Colors
# ----------------------------------------------------------

class LogColors:
    RESET = "\033[0m"

    GREY = "\033[90m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD_RED = "\033[31;1m"


class ColorFormatter(logging.Formatter):
    """
    Adds ANSI colors to console logs.
    """

    COLORS = {
        logging.DEBUG: LogColors.GREY,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.BOLD_RED,
    }

    def format(self, record):

        color = self.COLORS.get(record.levelno)

        fmt = (
            f"{color}"
            "[%(asctime)s]"
            "[%(levelname)s]"
            "[%(name)s]"
            " %(message)s"
            f"{LogColors.RESET}"
        )

        formatter = logging.Formatter(
            fmt=fmt,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        return formatter.format(record)


# ----------------------------------------------------------
# File Formatter
# ----------------------------------------------------------

FILE_FORMAT = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ----------------------------------------------------------
# Logger Factory
# ----------------------------------------------------------

def get_logger(
    name: str,
    log_file: str = "application.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a reusable logger.

    Parameters
    ----------
    name : str
        Logger name.

    log_file : str
        Output log filename.

    level : int
        Logging level.

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # ------------------------------------------------------
    # Console Handler
    # ------------------------------------------------------

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColorFormatter())

    # ------------------------------------------------------
    # File Handler
    # ------------------------------------------------------

    file_handler = RotatingFileHandler(
        LOG_DIR / log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setFormatter(FILE_FORMAT)
    file_handler.setLevel(level)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ----------------------------------------------------------
# Default Project Loggers
# ----------------------------------------------------------

train_logger = get_logger(
    name="TRAIN",
    log_file="training.log",
)

api_logger = get_logger(
    name="API",
    log_file="api.log",
)

evaluation_logger = get_logger(
    name="EVALUATION",
    log_file="evaluation.log",
)

prediction_logger = get_logger(
    name="PREDICTION",
    log_file="prediction.log",
)


# ----------------------------------------------------------
# Example Usage
# ----------------------------------------------------------

if __name__ == "__main__":

    train_logger.info("Training started.")

    train_logger.warning("Learning rate is very low.")

    train_logger.error("Dataset not found.")

    api_logger.info("API initialized.")

    prediction_logger.info("Prediction completed.")

    evaluation_logger.info("Evaluation finished.")