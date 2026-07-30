"""
LLM Abstraction Interface.

Defines the interface for text generation models, shielding the application 
from direct dependencies on the Google Gemini API.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    """
    Abstract Base Class for LLMs.
    
    Provides an interface to execute generation prompts with optional system 
    instructions (system prompts).
    """

    @abstractmethod
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Generate a text response for the given prompt and system instructions.

        Args:
            prompt: User-facing prompt or query.
            system_instruction: Guidelines or rules for the LLM behavior.

        Returns:
            The generated response string.
        """
        pass
