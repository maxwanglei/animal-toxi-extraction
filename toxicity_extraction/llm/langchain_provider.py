from typing import Optional

from langchain_meta import ChatMetaLlama

from .base import LLMProvider


class LangChainProvider(LLMProvider):
    """
    LangChain provider that uses ChatMetaLlama.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "Llama-4-Maverick-17B-128E-Instruct-FP8",
        base_url: Optional[str] = None,
    ):
        self.llm = ChatMetaLlama(model=model, api_key=api_key, base_url=base_url)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.llm.invoke(messages)
        return response.content
