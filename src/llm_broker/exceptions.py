from http import HTTPStatus


class LLMBrokerError(Exception):
    """Базовое исключение для всех ошибок пакета llm_broker."""


class LLMServiceError(LLMBrokerError):
    """Базовое исключение для ошибок LLM сервиса."""

    def __init__(self, message: str, status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class LLMServiceUnavailableError(LLMServiceError):
    """Исключение, возникающее когда LLM сервис недоступен."""

    def __init__(self, message: str = "LLM сервис недоступен"):
        super().__init__(
            message=message,
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )


class LLMServiceOverloadedError(LLMServiceError):
    """Исключение, возникающее когда LLM сервис перегружен."""

    def __init__(self, message: str = "LLM сервис перегружен"):
        super().__init__(
            message=message,
            status_code=HTTPStatus.TOO_MANY_REQUESTS,
        )


class LLMServiceHTTPError(LLMServiceError):
    """Исключение, возникающее при HTTP ошибках LLM сервиса."""

    def __init__(self, message: str, status_code: int):
        super().__init__(
            message=message,
            status_code=status_code,
        )


class LLMServiceGeneralError(LLMServiceError):
    """Исключение, возникающее при общих ошибках LLM сервиса."""

    def __init__(self, message: str = "Ошибка LLM сервиса"):
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


class CircuitBreakerOpenError(LLMBrokerError):
    """Raised when circuit breaker is open"""

    pass


class OverloadError(LLMBrokerError):
    """Raised when actor mailbox is full"""

    pass


class PoolShuttingDownError(LLMBrokerError):
    """Raised when pool is shutting down"""

    pass
