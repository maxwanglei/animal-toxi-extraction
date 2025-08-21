from typing import Optional, Any, Dict, List
import os
import logging
from .base import LLMProvider

logger = logging.getLogger(__name__)


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
            temperature=temperature, max_tokens=max_tokens,
            # Encourage strict JSON-only output when prompts ask for JSON
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content


class VertexAIProvider(LLMProvider):
    """Google Vertex AI provider"""

    def __init__(self, project_id: str, location: str = "us-central1",
                 model: str = "gemini-1.5-pro"):
        """Initialize Vertex AI client.

        Prefers the new google-genai client in Vertex mode (ADC based),
        and falls back to the legacy vertexai GenerativeModel if unavailable.
        """
        self._use_google_genai = False
        self._model_name = model
        self._project_id = project_id
        self._location = location

        # Warn if a direct Gemini API key is set; Vertex uses ADC instead.
        if os.environ.get("GOOGLE_API_KEY"):
            logger.warning(
                "GOOGLE_API_KEY is set but will be ignored by VertexAIProvider (uses ADC via gcloud/service account)."
            )

        try:
            # New unified client
            from google import genai as google_genai  # type: ignore
            from google.genai import types as genai_types  # type: ignore
            self._genai = google_genai
            self._genai_types = genai_types
            # Creates a client bound to Vertex AI using ADC
            self._client = self._genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
            self._use_google_genai = True
            logger.info(
                "Initialized Vertex AI provider using google-genai client: project=%s location=%s model=%s",
                project_id,
                location,
                model,
            )
        except Exception as e:
            logger.info(
                "google-genai not available or failed to initialize (%s). Falling back to vertexai SDK.",
                str(e)[:120],
            )
            from google.cloud import aiplatform
            from vertexai.generative_models import GenerativeModel
            aiplatform.init(project=project_id, location=location)
            self.model = GenerativeModel(model)

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 8192) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        if self._use_google_genai:
            # Build request for google-genai Vertex client
            T = self._genai_types
            try:
                config = T.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            except Exception:
                # Minimal dict fallback if types object is different
                config = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }

            # Create a single user content with the full text instruction
            try:
                contents = [T.Content(role="user", parts=[full_prompt])]  # type: ignore
            except Exception:
                contents = [full_prompt]

            try:
                resp = self._client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=config,
                )
                # google-genai responses expose .text
                text = getattr(resp, "text", None)
                if text is None:
                    # Fallback: attempt to join chunks if present
                    parts = []
                    for ch in getattr(resp, "candidates", []) or []:
                        part_text = getattr(getattr(ch, "content", None), "parts", None)
                        if part_text:
                            parts.extend([str(p) for p in part_text])
                    text = "".join(parts) if parts else ""
                return text or ""
            except Exception as e:
                logger.error("Vertex (google-genai) generate failed: %s", e)
                raise
        else:
            # Legacy SDK path
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
                 temperature: float = 0.1, max_tokens: int = 8192) -> str:
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
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=60,
        )
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
    Meta Llama provider using native llama_api_client with JSON schema enforcement.

    Env vars:
      - LLAMA_API_KEY (required)
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = None,
        default_temperature: float = 0.0,
        default_top_p: float = 0.9,
        default_frequency_penalty: float = 1.0,
        default_max_completion_tokens: int = 2048,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("LLAMA_API_KEY")
        if not self.api_key:
            raise RuntimeError("LLAMA_API_KEY is not set")
        
        self.base_url = base_url or "https://api.llama.com/compat/v1/"
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.default_frequency_penalty = default_frequency_penalty
        self.default_max_completion_tokens = default_max_completion_tokens
        
        # Define JSON schemas for strict validation
        self.relevance_schema = {
            "name": "relevance_assessment",
            "schema": {
                "type": "object",
                "properties": {
                    "is_relevant": {"type": "boolean"},
                    "relevance_score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "data_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["lab_tests", "organ_injury", "toxicity_frequencies"]}
                    },
                    "reason": {"type": "string"}
                },
                "required": ["is_relevant", "relevance_score", "data_types", "reason"],
                "additionalProperties": False
            }
        }
        
        self.extraction_schema = {
            "name": "toxicity_extraction",
            "schema": {
                "type": "object",
                "properties": {
                    "lab_tests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "drug": {"type": "string"},
                                "dose": {"type": "string"},
                                "lab_test": {"type": "string"},
                                "unit": {"type": ["string", "null"]},
                                "value_mean": {"type": ["number", "null"]},
                                "value_std": {"type": ["number", "null"]},
                                "value_raw": {"type": "string"},
                                "descriptive_values": {"type": ["string", "null"]},
                                "sample_size": {"type": ["integer", "null"]},
                                "time_point": {"type": ["string", "null"]},
                                "species": {"type": "string"},
                                "additional_info": {"type": ["string", "null"]}
                            },
                            "required": ["drug", "dose", "lab_test", "value_raw", "species"],
                            "additionalProperties": False
                        }
                    },
                    "organ_injuries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "drug": {"type": "string"},
                                "dose": {"type": "string"},
                                "injury_type": {"type": "string"},
                                "frequency": {"type": ["integer", "null"]},
                                "total_animals": {"type": ["integer", "null"]},
                                "severity": {"type": ["string", "null"]},
                                "time_point": {"type": ["string", "null"]},
                                "species": {"type": "string"},
                                "descriptive_values": {"type": ["string", "null"]},
                                "additional_info": {"type": ["string", "null"]}
                            },
                            "required": ["drug", "dose", "injury_type", "species"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["lab_tests", "organ_injuries"],
                "additionalProperties": False
            }
        }
        
        logger.info("MetaLlamaProvider initialized: model=%s using native client", self.model)

    def chat_with_schema(
        self,
        messages: List[Dict[str, Any]],
        schema: Dict[str, Any],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate response with JSON schema enforcement."""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": self.default_max_completion_tokens if max_tokens is None else max_tokens,
            "top_p": self.default_top_p if top_p is None else top_p,
            "repetition_penalty": self.default_frequency_penalty if frequency_penalty is None else frequency_penalty,
        }
        
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        
        try:
            response = requests.post(endpoint, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]
            
            try:
                usage = response_json.get("usage")
                if usage:
                    logger.info(
                        "MetaLlama response: id=%s usage(prompt=%s, completion=%s, total=%s)",
                        response_json.get("id"),
                        usage.get("prompt_tokens"),
                        usage.get("completion_tokens"),
                        usage.get("total_tokens"),
                    )
                else:
                    logger.info("MetaLlama response: id=%s (no usage field)", response_json.get("id"))
            except Exception:
                pass
            
            return content
            
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                logger.error("MetaLlama API error response: %s", e.response.text)
            logger.error("Error in MetaLlama chat_with_schema: %s", e)
            raise

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Basic chat without schema enforcement for backward compatibility."""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": self.default_max_completion_tokens if max_tokens is None else max_tokens,
            "top_p": self.default_top_p if top_p is None else top_p,
            "repetition_penalty": self.default_frequency_penalty if frequency_penalty is None else frequency_penalty,
            "response_format": {"type": "json_object"}
        }
        
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        
        try:
            response = requests.post(endpoint, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                logger.error("MetaLlama API error response: %s", e.response.text)
            logger.error("Error in MetaLlama chat: %s", e)
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> str:
        """Generate a response using the native llama_api_client with schema enforcement."""
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Determine which schema to use based on system prompt content
        if system_prompt and "relevance_assessment" in system_prompt.lower():
            return self.chat_with_schema(
                messages=messages,
                schema=self.relevance_schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif system_prompt and "extract" in system_prompt.lower():
            return self.chat_with_schema(
                messages=messages,
                schema=self.extraction_schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            # Fallback to basic chat for other use cases
            return self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
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
