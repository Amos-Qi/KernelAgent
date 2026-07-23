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
import time
from typing import Any
import logging
from .base import BaseProvider, LLMResponse
from .env_config import configure_proxy_environment

try:
    import httpx
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    httpx = None


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
            # Single completions stream (see _stream_and_trace), so this
            # timeout bounds *stalls* — connect, time-to-first-token (server
            # queue + prefill), and gaps between chunks — not total duration.
            # An actively generating model is never cut off mid-thought; a
            # dead connection or stuck queue fails within one stall budget.
            stall_s = float(os.environ.get("KERNELAGENT_LLM_TIMEOUT_S", "900"))
            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": httpx.Timeout(stall_s, connect=60.0),
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
        content, finish, trace = self._stream_and_trace(api_params, model_name)

        if (
            content is None
            and finish == "length"
            and model_name.startswith("glm")
            and not self._thinking_disabled(api_params)
        ):
            logging.getLogger(__name__).warning(
                "%s: reasoning consumed the completion budget; retrying with "
                "thinking disabled",
                model_name,
            )
            retry_params = dict(api_params)
            retry_params["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
            content, finish, trace = self._stream_and_trace(retry_params, model_name)

        if content is None:
            raise RuntimeError(
                f"{self.name} returned no content for {model_name} "
                f"(finish_reason={finish}); trace: {trace}"
            )

        return LLMResponse(
            content=content,
            model=model_name,
            provider=self.name,
        )

    def _stream_and_trace(
        self, api_params: dict[str, Any], model_name: str
    ) -> tuple[str | None, str | None, str]:
        """Stream one chat completion, mirroring thinking/answer tokens to a
        trace file as they arrive.

        Makes long reasoning calls observable (tail -f the newest file in the
        trace dir) and makes failures diagnosable: the trace distinguishes
        server-queue silence (no first chunk) from live decoding (reasoning
        text flowing) from a mid-stream stall. Returns
        (content or None, finish_reason, trace_path).
        """
        trace_dir = os.environ.get(
            "KERNELAGENT_LLM_TRACE_DIR",
            os.path.expanduser("~/.kernelagent/llm_traces"),
        )
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(
            trace_dir,
            f"{time.strftime('%Y%m%d_%H%M%S')}_{model_name}_pid{os.getpid()}.trace",
        )

        params = dict(api_params)
        params["stream"] = True
        reasoning_chars = 0
        content_parts: list[str] = []
        finish: str | None = None
        t0 = time.time()
        first_chunk_at: float | None = None
        last_flush = t0

        with open(trace_path, "w") as tf:
            tf.write(
                f"# model={model_name} max_tokens={params.get('max_tokens')} "
                f"extra_body={params.get('extra_body')} started={time.ctime(t0)}\n"
            )
            tf.flush()
            try:
                stream = self.client.chat.completions.create(**params)
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if first_chunk_at is None:
                        first_chunk_at = time.time()
                        tf.write(
                            f"# [first-chunk] t={first_chunk_at - t0:.1f}s "
                            "(server queue + prefill)\n"
                        )
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning is None:
                        extra = getattr(delta, "model_extra", None) or {}
                        reasoning = extra.get("reasoning") or extra.get(
                            "reasoning_content"
                        )
                    if reasoning:
                        reasoning_chars += len(reasoning)
                        tf.write(reasoning)
                    if delta.content:
                        content_parts.append(delta.content)
                        tf.write(delta.content)
                    if choice.finish_reason:
                        finish = choice.finish_reason
                    now = time.time()
                    if now - last_flush >= 30:
                        tf.write(
                            f"\n# [progress] t={now - t0:.0f}s "
                            f"reasoning_chars={reasoning_chars} "
                            f"answer_chars={sum(len(p) for p in content_parts)}\n"
                        )
                        tf.flush()
                        last_flush = now
            except Exception as e:
                elapsed = time.time() - t0
                answer_chars = sum(len(p) for p in content_parts)
                tf.write(
                    f"\n# [aborted] t={elapsed:.0f}s {type(e).__name__}: {e} | "
                    f"first_chunk={'never' if first_chunk_at is None else f'{first_chunk_at - t0:.1f}s'} "
                    f"reasoning_chars={reasoning_chars} answer_chars={answer_chars}\n"
                )
                raise RuntimeError(
                    f"{self.name} stream aborted after {elapsed:.0f}s "
                    f"({type(e).__name__}); first chunk "
                    f"{'never arrived' if first_chunk_at is None else f'after {first_chunk_at - t0:.1f}s'}, "
                    f"{reasoning_chars} reasoning chars and {answer_chars} answer "
                    f"chars streamed; trace: {trace_path}"
                ) from e

            elapsed = time.time() - t0
            answer_chars = sum(len(p) for p in content_parts)
            tf.write(
                f"\n# [done] t={elapsed:.0f}s finish={finish} "
                f"reasoning_chars={reasoning_chars} answer_chars={answer_chars}\n"
            )

        content = "".join(content_parts) or None
        return content, finish, trace_path

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
            # callers' answer-sized asks (16-24k) would strangle it: grant
            # KERNELAGENT_GLM_THINKING_BUDGET (default 60k) capped by the
            # model limit. Overflow triggers the thinking-off retry below, so
            # a non-converging ramble costs one bounded attempt.
            budget = int(os.environ.get("KERNELAGENT_GLM_THINKING_BUDGET", "60000"))
            max_tokens_value = min(budget, self.get_max_tokens_limit(model_name))
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

        if model_name.startswith("glm"):
            if glm_thinking:
                # GLM-5.2 API contract (docs.z.ai migrate-to-glm-new):
                # thinking={"type":"enabled"} + top-level reasoning_effort in
                # {"high","max"}, default max. Verified on the Unity endpoint:
                # "high" halves reasoning volume and is far more consistent
                # than the "max" default, so it's our default.
                effort = os.environ.get("KERNELAGENT_GLM_REASONING_EFFORT", "high")
                params["extra_body"] = {
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": effort,
                }
            else:
                # Thinking disabled: tell the vLLM chat template so the budget
                # goes straight to the answer. (A length-truncated thinking
                # call is retried once with this same setting as a safety net.)
                params["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }

        return params

    @staticmethod
    def _thinking_disabled(api_params: dict[str, Any]) -> bool:
        """True when the request already tells the chat template not to think."""
        extra = api_params.get("extra_body") or {}
        ctk = extra.get("chat_template_kwargs") or {}
        return ctk.get("enable_thinking") is False

    def _create_with_thinking_fallback(self, api_params: dict[str, Any]):
        """Run chat.completions.create; if a thinking model consumed the whole
        budget reasoning (finish_reason=length, content=None), retry once with
        thinking disabled rather than failing the call."""
        response = self.client.chat.completions.create(**api_params)
        retriable = (
            str(api_params.get("model", "")).startswith("glm")
            and not self._thinking_disabled(api_params)
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
