from typing import Optional

from src.config.config import settings
from src.llm.base import BaseLLM
from src.llm.gemini import GeminiLLM
from src.llm.groq import GroqLLM


def get_llm(provider: Optional[str] = None) -> BaseLLM:
    """
    Factory function to retrieve the configured LLM implementation.
    
    If provider is 'groq', returns GroqLLM.
    Otherwise (or if default), returns GeminiLLM.
    """
    if provider == "groq":
        return GroqLLM()
    return GeminiLLM()
