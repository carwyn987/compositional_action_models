"""Logging setup for entry points. Library modules only call logging.getLogger(__name__)."""

import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Send log records from the cam package to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
    logger = logging.getLogger("cam")
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False
