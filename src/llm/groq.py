from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config.config import settings
from src.llm.base import BaseLLM


class GroqLLM(BaseLLM):
    """
    Groq API wrapper for text generation using langchain_openai.
    """

    def __init__(self) -> None:
        """Initialize the Groq client using config settings."""
        self._llm = ChatOpenAI(
            model=settings.groq_model,
            openai_api_key=settings.groq_api_key,
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=settings.temperature
        )

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Generate text using Groq with prompt and system prompts.
        """
        messages: List[BaseMessage] = []
        
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))
            
        messages.append(HumanMessage(content=prompt))
        
        # Invoke LLM API
        response = self._llm.invoke(messages)
        
        return str(response.content)
