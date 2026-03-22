# ==== Standard Imports ====
import logging
from pathlib import Path


def setup_logger(log_path: Path, log_filename: str, logger_name: str) -> logging.Logger:
    """
    Sets up a logging system for a specific area. It could be for
    training, preprocessing, or some sort.

    Args:
    :param log_path: The output path of the log file.
    :param log_filename: The filename for the log file.
    :param logger_name: The name of the logger.
    :return: A logger object.
    """
    log_file = log_path / log_filename
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # create a new logger if not in the handlers
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_file)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger