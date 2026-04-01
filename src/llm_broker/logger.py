import sys
from typing import Any

from loguru import logger


class BrokerLogger:
    """Централизованный логгер для пакета llm_broker на основе loguru."""

    _configured = False

    @classmethod
    def configure(
        cls,
        level: str = "INFO",
    ) -> None:
        """
        Добавляет обработчик loguru для llm_broker.

        Не удаляет существующие обработчики приложения-хоста.
        Если нужно заменить все обработчики — вызови ``logger.remove()``
        до вызова этого метода.

        Args:
            level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if cls._configured:
            return

        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
            "<level>{message}</level>"
        )

        logger.add(
            sys.stderr,
            format=console_format,
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

        cls._configured = True

    @classmethod
    def get_logger(cls, name: str | None = None) -> Any:
        """
        Возвращает настроенный логгер.

        Args:
            name: Имя модуля для логирования (опционально)

        Returns:
            Экземпляр loguru logger
        """
        if not cls._configured:
            cls.configure()

        if name:
            return logger.bind(name=name)
        return logger

    @classmethod
    def bind_context(
        cls,
        pool_id: str | None = None,
        actor_id: str | None = None,
        request_id: str | None = None,
    ) -> Any:
        """
        Создает логгер с привязанным контекстом.

        Args:
            pool_id: ID пула акторов
            actor_id: ID актора
            request_id: ID запроса

        Returns:
            Логгер с привязанным контекстом
        """
        context = {}
        if pool_id:
            context["pool_id"] = pool_id
        if actor_id:
            context["actor_id"] = actor_id
        if request_id:
            context["request_id"] = request_id

        return logger.bind(**context)
