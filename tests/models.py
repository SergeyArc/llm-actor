from pydantic import BaseModel


class User(BaseModel):
    """Sample Pydantic model for structured-output tests."""

    name: str
    age: int
