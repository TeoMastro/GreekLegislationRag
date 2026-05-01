from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"
