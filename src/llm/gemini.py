"""
Google Gemini LLM Implementation.

Concrete implementation of the BaseLLM interface using ChatGoogleGenAI 
to interact with the Google Gemini API (defaulting to gemini-2.5-flash).
"""

from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config.config import settings
from src.llm.base import BaseLLM


class GeminiLLM(BaseLLM):
    """
    Google Gemini API wrapper for text generation.
    
    Adheres to clean architecture by wrapping the LangChain ChatGoogleGenerativeAI client.
    """

    def __init__(self) -> None:
        """Initialize the Gemini client using config settings."""
        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.temperature
        )

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Generate text using ChatGoogleGenAI with prompt and system prompts.

        Args:
            prompt: User-facing prompt.
            system_instruction: Optional system instruction governing model constraints.

        Returns:
            The generated response string from Gemini.
        """
        messages: List[BaseMessage] = []
        
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))
            
        messages.append(HumanMessage(content=prompt))
        
        # Invoke Gemini API
        response = self._llm.invoke(messages)
        
        # Ensure returned response is a string
        return str(response.content)
