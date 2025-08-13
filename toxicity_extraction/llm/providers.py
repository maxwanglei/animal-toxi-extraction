from typing import Optional, Any, Dict, List
import os

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 2000) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens
        )
        return response.choices[0].message.content


class VertexAIProvider(LLMProvider):
    """Google Vertex AI provider"""

    def __init__(self, project_id: str, location: str = "us-central1",
                 model: str = "gemini-1.5-pro"):
        from google.cloud import aiplatform
        from vertexai.generative_models import GenerativeModel
        aiplatform.init(project=project_id, location=location)
        self.model = GenerativeModel(model)

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 2000) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = self.model.generate_content(
            full_prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
        )
        return response.text


class GeminiProvider(LLMProvider):
    """Google Gemini API provider"""

    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 2000) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = self.model.generate_content(
            full_prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
        )
        return response.text


class LlamaAPIProvider(LLMProvider):
    """Meta Llama API provider"""

    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.2-70B"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.llama-api.com"

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 2000) -> str:
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": self.model, "messages": [], "temperature": temperature, "max_tokens": max_tokens}
        if system_prompt:
            data["messages"].append({"role": "system", "content": system_prompt})
        data["messages"].append({"role": "user", "content": prompt})
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class LocalLLMProvider(LLMProvider):
    """Local LLM provider for supercomputer deployment"""

    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.device = device
        self.torch = torch

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 2000) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(full_prompt):].strip()


try:
    # OpenAI SDK (v1) with compat base_url
    from openai import OpenAI
except Exception:
    OpenAI = None


class MetaLlamaProvider(LLMProvider):
    """
    Meta Llama provider via OpenAI-compatible Chat Completions API.

    Env vars:
      - LLAMA_API_KEY (required)
      - LLAMA_BASE_URL (optional; defaults to https://api.llama.com/compat/v1/)
    """
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        default_temperature: float = 0.6,
        default_top_p: float = 0.9,
        default_frequency_penalty: float = 1.0,
        default_max_completion_tokens: int = 2048,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package not installed. Run: pip install openai")
        self.model = model
        self.api_key = api_key or os.environ.get("LLAMA_API_KEY")
        if not self.api_key:
            raise RuntimeError("LLAMA_API_KEY is not set")
        self.base_url = base_url or os.environ.get("LLAMA_BASE_URL", "https://api.llama.com/compat/v1/")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.default_frequency_penalty = default_frequency_penalty
        self.default_max_completion_tokens = default_max_completion_tokens

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        # messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        resp = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.default_temperature if temperature is None else temperature,
            # Use OpenAI-compatible parameter name
            max_tokens=self.default_max_completion_tokens if max_tokens is None else max_tokens,
            top_p=self.default_top_p if top_p is None else top_p,
            frequency_penalty=self.default_frequency_penalty if frequency_penalty is None else frequency_penalty,
        )
        # Return assistant text
        return resp.choices[0].message.content

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> str:
        """Generate a response using the OpenAI-compatible chat completions API.

        This wraps the chat() method to satisfy the LLMProvider abstract interface.
        """
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(
            messages=messages,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_completion_tokens,
        )


def make_provider(name: str, **kwargs) -> LLMProvider:
    """
    Factory to create a provider by name.
    """
    if name == "openai":
        return OpenAIProvider(api_key=kwargs["api_key"], model=kwargs["model"])
    if name == "vertex-ai":
        return VertexAIProvider(project_id=kwargs["project_id"], location=kwargs.get("location", "us-central1"), model=kwargs["model"])
    if name == "gemini":
        return GeminiProvider(api_key=kwargs["api_key"], model=kwargs["model"])
    if name in ("meta-llama", "llama", "llama-meta"):
        # Expected kwargs: model, api_key?, base_url?, temperature?, top_p?, frequency_penalty?, max_tokens?
        return MetaLlamaProvider(
            model=kwargs.get("model"),
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url"),
            default_temperature=kwargs.get("temperature", 0.6),
            default_top_p=kwargs.get("top_p", 0.9),
            default_frequency_penalty=kwargs.get("frequency_penalty", 1.0),
            default_max_completion_tokens=kwargs.get("max_tokens", 2048),
        )
    # For backward compatibility with existing code
    if name == "llama-api":
        from warnings import warn
        warn("llama-api provider is deprecated, use meta-llama instead", DeprecationWarning, stacklevel=2)
        return LlamaAPIProvider(api_key=kwargs["api_key"], model=kwargs["model"])
    if name == "local":
        return LocalLLMProvider(model_path=kwargs["model_path"], device=kwargs.get("device", "cuda"))

    raise ValueError(f"Unknown provider: {name}")
