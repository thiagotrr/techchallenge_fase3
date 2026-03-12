import logging
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LOGGER_NAME = "techchallenge_fase3"
_LOGGER: logging.Logger | None = None


def _get_log_file_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"log_record_{timestamp}.log"


def setup_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    """Configura e retorna um logger que escreve em arquivo com timestamp."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d-%m-%Y %H:%M:%S",
        )

        file_handler = logging.FileHandler(_get_log_file_path(), encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def get_logger() -> logging.Logger:
    """Retorna o logger singleton para a aplicação."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = setup_logger()
    return _LOGGER

