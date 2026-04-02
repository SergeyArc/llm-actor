from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMBrokerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=True, extra="ignore")

    LLM_FAILURE_THRESHOLD: int = 5
    LLM_RECOVERY_TIMEOUT: float = 60.0
    LLM_BATCH_SIZE: int = 4
    LLM_BATCH_TIMEOUT: float = 0.01
    LLM_MAX_CONCURRENT: int = 25
    LLM_MAX_QUEUE_SIZE: int = 1000
    LLM_MAX_CONSECUTIVE_FAILURES: int = 2
    LLM_NUM_ACTORS: int = 10
    LLM_MAX_RESTARTS: int = 10
    LLM_RESTART_WINDOW: float = 60.0
    LLM_GRACEFUL_SHUTDOWN_TIMEOUT: float = 30.0
    LLM_RETRY_MAX_ATTEMPTS: int = 3
    LLM_RETRY_BASE_BACKOFF: float = 1.0
    LLM_RETRY_BACKOFF_CAP: float = 16.0
    LLM_VALIDATION_RETRY_MAX_ATTEMPTS: int = 5
    LLM_TOOL_EXECUTION_TIMEOUT: float = 30.0
    LLM_TOOL_MAX_ITERATIONS: int = 10

    @field_validator(
        "LLM_NUM_ACTORS",
        "LLM_BATCH_SIZE",
        "LLM_MAX_CONCURRENT",
        "LLM_MAX_QUEUE_SIZE",
        "LLM_RETRY_MAX_ATTEMPTS",
        "LLM_VALIDATION_RETRY_MAX_ATTEMPTS",
        "LLM_FAILURE_THRESHOLD",
        "LLM_MAX_CONSECUTIVE_FAILURES",
        "LLM_MAX_RESTARTS",
        "LLM_TOOL_MAX_ITERATIONS",
    )
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Value must be >= 1, got {v}")
        return v

    @field_validator(
        "LLM_BATCH_TIMEOUT",
        "LLM_RECOVERY_TIMEOUT",
        "LLM_GRACEFUL_SHUTDOWN_TIMEOUT",
        "LLM_TOOL_EXECUTION_TIMEOUT",
    )
    @classmethod
    def must_be_positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Value must be > 0, got {v}")
        return v

