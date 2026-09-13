import rospy
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseTool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with provided arguments.
        Returns a dictionary containing the result.
        """
        pass

    def get_definition(self) -> Dict[str, Any]:
        """
        Returns a JSON-schema like definition for the LLM.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema()
        }

    @abstractmethod
    def _get_parameters_schema(self) -> Dict[str, Any]:
        pass

