# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base provider for OpenAI-compatible APIs."""

import os
from typing import Any
import logging
from .base import BaseProvider, LLMResponse
from .env_config import configure_proxy_environment

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class OpenAICompatibleProvider(BaseProvider):
    """Base provider for OpenAI-compatible APIs."""

    def __init__(self, api_key_env: str, base_url: str | None = None):
        self.api_key_env = api_key_env
        self.base_url = base_url
        self._original_proxy_env = None
        super().__init__()

    def _initialize_client(self) -> None:
        """Initialize OpenAI-compatible client."""
        if not OPENAI_AVAILABLE:
            return

        api_key = self._get_api_key(self.api_key_env)
        if api_key:
            # Configure proxy using centralized utility function
            self._original_proxy_env = configure_proxy_environment()

            # Initialize client (proxy configured via environment variables).
            # The SDK's default 600s request timeout is far too short for
            # reasoning models: a 150k-token thinking budget at ~25 tok/s can
            # legitimately stream for ~100 minutes. One long attempt beats
            # the SDK's silent timeout->retry loop.
            timeout_s = float(os.environ.get("KERNELAGENT_LLM_TIMEOUT_S", "7200"))
            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": timeout_s,
                "max_retries": 1,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = OpenAI(**client_kwargs)

    def get_response(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Get single response."""
        if not self.is_available():
            raise RuntimeError(f"{self.name} client not available")

        api_params = self._build_api_params(model_name, messages, **kwargs)
        response = self._create_with_thinking_fallback(api_params)
        logging.getLogger(__name__).info(
            "OpenAI chat response (single): %s",
            getattr(response, "model_dump", lambda: str(response))(),
        )

        return LLMResponse(
            content=self._require_content(response.choices[0], model_name),
            model=model_name,
            provider=self.name,
            usage=response.usage.dict()
            if hasattr(response, "usage") and response.usage
            else None,
        )

    def get_multiple_responses(
        self, model_name: str, messages: list[dict[str, str]], n: int = 1, **kwargs
    ) -> list[LLMResponse]:
        """Get multiple responses using n parameter."""
        if not self.is_available():
            raise RuntimeError(f"{self.name} client not available")

        api_params = self._build_api_params(model_name, messages, n=n, **kwargs)
        response = self._create_with_thinking_fallback(api_params)
        logging.getLogger(__name__).info(
            "OpenAI chat response (multi): %s",
            getattr(response, "model_dump", lambda: str(response))(),
        )

        return [
            LLMResponse(
                content=self._require_content(choice, model_name),
                model=model_name,
                provider=self.name,
                usage=response.usage.dict()
                if hasattr(response, "usage") and response.usage
                else None,
            )
            for choice in response.choices
        ]

    def _require_content(self, choice: Any, model_name: str) -> str:
        """Return the choice's content, failing loudly when it is missing.

        Reasoning models on OpenAI-compatible endpoints return content=None
        when the completion budget is exhausted mid-thought
        (finish_reason='length'); surfacing that here beats a downstream
        TypeError from writing None to a log file.
        """
        content = choice.message.content
        if content is None:
            finish = getattr(choice, "finish_reason", "unknown")
            raise RuntimeError(
                f"{self.name} returned no content for {model_name} "
                f"(finish_reason={finish}). If finish_reason is 'length', the "
                "max_tokens budget was consumed by reasoning before the "
                "answer — raise get_max_tokens_limit for this model."
            )
        return content

    def _build_api_params(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> dict[str, Any]:
        """Build API parameters for OpenAI-compatible call."""
        params = {
            "model": model_name,
            "messages": messages,
        }

        # GPT-5 and o-series models pin their own sampling behaviour
        if not (model_name.startswith("gpt-5") or model_name.startswith("o")):
            params["temperature"] = kwargs.get("temperature", 0.7)

        # GLM thinking is on unless KERNELAGENT_GLM_THINKING=off.
        glm_thinking = (
            model_name.startswith("glm")
            and os.environ.get("KERNELAGENT_GLM_THINKING", "on").lower() != "off"
        )

        # Use max_completion_tokens for newer models like GPT-5, fallback to max_tokens
        if glm_thinking:
            # Chain-of-thought burns completion budget before the answer, so
            # callers' answer-sized asks (16-24k) would strangle it; grant the
            # model limit instead.
            max_tokens_value = self.get_max_tokens_limit(model_name)
        else:
            max_tokens_value = min(
                kwargs.get("max_tokens", 8192), self.get_max_tokens_limit(model_name)
            )
        if model_name.startswith("gpt-5") or model_name.startswith("o"):
            params["max_completion_tokens"] = max_tokens_value
        else:
            params["max_tokens"] = max_tokens_value

        # Add n parameter if specified
        if "n" in kwargs:
            params["n"] = kwargs["n"]

        # Auto-enable high reasoning for GPT-5
        if model_name.startswith("gpt-5"):
            params["reasoning_effort"] = "high"
        elif kwargs.get("high_reasoning_effort") and model_name.startswith(
            ("o3", "o1")
        ):
            params["reasoning_effort"] = "high"

        # Thinking disabled: tell the vLLM chat template so the budget goes
        # straight to the answer. (When thinking is on, a length-truncated
        # call is still retried once without thinking as a safety net.)
        if model_name.startswith("glm") and not glm_thinking:
            params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        return params

    def _create_with_thinking_fallback(self, api_params: dict[str, Any]):
        """Run chat.completions.create; if a thinking model consumed the whole
        budget reasoning (finish_reason=length, content=None), retry once with
        thinking disabled rather than failing the call."""
        response = self.client.chat.completions.create(**api_params)
        retriable = (
            str(api_params.get("model", "")).startswith("glm")
            and "extra_body" not in api_params
            and any(
                c.message.content is None
                and getattr(c, "finish_reason", "") == "length"
                for c in response.choices
            )
        )
        if retriable:
            logging.getLogger(__name__).warning(
                "%s: reasoning consumed the completion budget; retrying with "
                "thinking disabled",
                api_params.get("model"),
            )
            retry_params = dict(api_params)
            retry_params["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
            response = self.client.chat.completions.create(**retry_params)
        return response

    def is_available(self) -> bool:
        """Check if provider is available."""
        return OPENAI_AVAILABLE and self.client is not None

    def supports_multiple_completions(self) -> bool:
        """OpenAI-compatible APIs support native multiple completions."""
        return True
