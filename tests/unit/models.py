from pydantic import BaseModel


class User(BaseModel):
    """Тестовая Pydantic модель для проверки response_model."""

    name: str
    age: int
