"""Logger configuration module."""

import logging


def setup_logger() -> logging.Logger:
    """Set up and configure a basic logger instance.

    Returns:
        logging.Logger: Configured logger instance.
    """
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if _logger.handlers:
        return _logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    _logger.addHandler(console_handler)

    # Prevent propagation to avoid duplicate messages from parent loggers
    _logger.propagate = False

    return _logger


logger = setup_logger()
