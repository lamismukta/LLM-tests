"""OpenAI LLM provider implementation."""
import os
from typing import Dict, Any
from openai import AsyncOpenAI
from .base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """OpenAI provider for GPT models."""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 2000):
        super().__init__(model, temperature, max_tokens)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response using OpenAI API."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert CV analyst with deep knowledge of recruitment and talent assessment."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            metadata={"finish_reason": response.choices[0].finish_reason}
        )
    
    def get_provider_name(self) -> str:
        return "openai"

