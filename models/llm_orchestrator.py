"""
LLM call orchestration layer
Unified encapsulation of different LLM provider API calls
"""

import asyncio
from typing import Optional, Dict, Any, List
import json
from abc import ABC, abstractmethod

from .llm_configs import ModelConfig, ModelProvider, TaskType


class BaseLLMClient(ABC):
    """LLM client base class"""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text"""
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate structured Output (JSON)"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=config.api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        return response.choices[0].message.content

    async def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate JSON format Output"""
        full_prompt = f"{prompt}\n\nPlease output result in JSON format."
        response = await self.generate(full_prompt, system_prompt)

        # Try to parse JSON
        try:
            # Extract JSON part (may be wrapped in markdown)
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parse failed: {e}\nOriginal response: {response}")
            return {"error": "JSON parse failed", "raw_response": response}


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=config.api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            response = await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            error_msg = str(e)
            # Check for common API errors
            if "credit balance is too low" in error_msg or "insufficient credits" in error_msg.lower():
                raise RuntimeError(
                    "API余额不足。请前往 https://console.anthropic.com/settings/billing 充值。\n"
                    "Insufficient API credits. Please visit https://console.anthropic.com/settings/billing to add credits."
                ) from e
            elif "authentication" in error_msg.lower() or "invalid" in error_msg.lower() and "api" in error_msg.lower():
                raise RuntimeError(
                    "API密钥无效或未设置。请检查 .env 文件中的 ANTHROPIC_API_KEY。\n"
                    "Invalid or missing API key. Please check ANTHROPIC_API_KEY in .env file."
                ) from e
            else:
                raise

    async def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        # generate() already handles API errors, so just call it
        full_prompt = f"{prompt}\n\nPlease output result in JSON format, without any other text."
        response = await self.generate(full_prompt, system_prompt)

        try:
            # Claude typically returns JSON directly
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parse failed: {e}\nOriginal response: {response}")
            return {"error": "JSON parse failed", "raw_response": response}


class LocalLLMClient(BaseLLMClient):
    """Local LLM client (like Llama, Mistral etc)"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # This can integrate with ollama, llama.cpp etc local inference engines
        print("Warning: local model support pending implementation, currently using mock pattern")

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # TODO: implement local model call
        await asyncio.sleep(0.1)  # simulate inference time
        return f"[local model response] Simulated reply to input"

    async def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        return {"status": "local model pending implementation", "mock_response": True}


class LLMOrchestrator:
    """
    LLM Orchestrator
    Unified interface, automatically route to appropriate model
    """

    def __init__(self):
        self._clients: Dict[str, BaseLLMClient] = {}

    def _get_client(self, config: ModelConfig) -> BaseLLMClient:
        """Get or create LLM client"""
        cache_key = f"{config.provider.value}_{config.model_name}"

        if cache_key not in self._clients:
            if config.provider == ModelProvider.OPENAI:
                self._clients[cache_key] = OpenAIClient(config)
            elif config.provider == ModelProvider.ANTHROPIC:
                self._clients[cache_key] = AnthropicClient(config)
            elif config.provider == ModelProvider.LOCAL:
                self._clients[cache_key] = LocalLLMClient(config)
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")

        return self._clients[cache_key]

    async def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text response

        Args:
            prompt: user prompt
            config: Model configuration
            system_prompt: system prompt
        """
        client = self._get_client(config)
        return await client.generate(prompt, system_prompt)

    async def generate_structured(
        self,
        prompt: str,
        config: ModelConfig,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON response

        Args:
            prompt: user prompt
            config: Model configuration
            system_prompt: system prompt
        """
        client = self._get_client(config)
        return await client.generate_structured(prompt, system_prompt)

    async def batch_generate(
        self,
        prompts: List[str],
        config: ModelConfig,
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Batch generate (concurrent)

        Args:
            prompts: Multiple prompts
            config: Model configuration
            system_prompt: system prompt
        """
        tasks = [self.generate(p, config, system_prompt) for p in prompts]
        return await asyncio.gather(*tasks)


# Global singleton
_orchestrator_instance = None


def get_orchestrator() -> LLMOrchestrator:
    """Get global LLM Orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = LLMOrchestrator()
    return _orchestrator_instance
