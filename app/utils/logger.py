"""
Настройка структурированного логирования.
"""
import logging
import sys
from typing import Optional

from pythonjsonlogger import jsonlogger


def setup_logging(log_level: str = "INFO"):
    """
    Настройка логирования с JSON форматом.

    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
    """
    # Создаем логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Удаляем существующие хендлеры
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Создаем JSON форматтер
    formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s %(filename)s %(lineno)d',
        json_ensure_ascii=False
    )

    # Консольный хендлер
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Настройка логов для внешних библиотек
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info(
        "Logging configured",
        extra={
            "log_level": log_level,
            "formatter": "json"
        }
    )