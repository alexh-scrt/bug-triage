"""LLM abstraction layer for bug_triage.

Provides a unified interface for calling OpenAI and Anthropic APIs with
exponential-backoff retry logic and Jinja2-based prompt templating.  All
public functions and classes in this module are designed to be used by the
triage engine and other pipeline stages.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import anthropic
import openai
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from bug_triage.models import LLMProvider

logger = logging.getLogger(__name__)

# Directory containing Jinja2 prompt templates bundled with the package.
_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Default models per provider.
_DEFAULT_MODELS: dict[str, str] = {
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
}

# Retry configuration.
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds
_RETRY_BACKOFF = 2.0


class LLMError(Exception):
    """Raised when an LLM API call fails after exhausting all retries."""


class PromptTemplateError(Exception):
    """Raised when a Jinja2 prompt template cannot be loaded or rendered."""


class PromptRenderer:
    """Renders Jinja2 prompt templates from the bundled prompts directory.

    Args:
        prompts_dir: Directory containing ``.j2`` template files.  Defaults to
            the ``bug_triage/prompts/`` directory bundled with the package.
    """

    def __init__(self, prompts_dir: Optional[Path] = None) -> None:
        self._prompts_dir = prompts_dir or _PROMPTS_DIR
        self._env = Environment(
            loader=FileSystemLoader(str(self._prompts_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_name: str, **context: Any) -> str:
        """Render a named Jinja2 template with the supplied context variables.

        Args:
            template_name: Filename of the template (e.g. ``triage_prompt.j2``).
            **context: Template variables passed to the Jinja2 render call.

        Returns:
            The rendered prompt string.

        Raises:
            PromptTemplateError: If the template file is not found or fails to
                render.
        """
        try:
            template = self._env.get_template(template_name)
        except TemplateNotFound as exc:
            raise PromptTemplateError(
                f"Prompt template '{template_name}' not found in '{self._prompts_dir}'."
            ) from exc
        except Exception as exc:
            raise PromptTemplateError(
                f"Failed to load prompt template '{template_name}': {exc}"
            ) from exc

        try:
            rendered = template.render(**context)
        except Exception as exc:
            raise PromptTemplateError(
                f"Failed to render prompt template '{template_name}': {exc}"
            ) from exc

        return rendered


class LLMClient:
    """Unified client for OpenAI and Anthropic LLM APIs.

    Supports automatic exponential-backoff retries on transient API errors
    and Jinja2 prompt templating via a bundled :class:`PromptRenderer`.

    Args:
        provider: The LLM provider to use (``openai`` or ``anthropic``).
        model: Model identifier to use.  Defaults to the provider's default.
        api_key: API key for the provider.  If ``None``, the client reads from
            the standard environment variable (``OPENAI_API_KEY`` or
            ``ANTHROPIC_API_KEY``).
        max_tokens: Maximum number of tokens in the LLM response.
        temperature: Sampling temperature for the LLM.  Lower values produce
            more deterministic outputs.
        max_retries: Number of retry attempts on transient failures.
        prompts_dir: Optional custom directory for Jinja2 prompt templates.
    """

    def __init__(
        self,
        provider: LLMProvider | str = LLMProvider.OPENAI,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        max_retries: int = _MAX_RETRIES,
        prompts_dir: Optional[Path] = None,
    ) -> None:
        self.provider = LLMProvider(provider)
        self.model = model or _DEFAULT_MODELS[self.provider]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.renderer = PromptRenderer(prompts_dir=prompts_dir)

        if self.provider == LLMProvider.OPENAI:
            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            self._openai_client = openai.OpenAI(**kwargs)
            self._anthropic_client = None
        else:
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            self._anthropic_client = anthropic.Anthropic(**kwargs)
            self._openai_client = None

        logger.info(
            "LLMClient initialised: provider=%s model=%s", self.provider, self.model
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a completion request to the configured LLM provider.

        Retries up to ``max_retries`` times on transient API errors using
        exponential backoff.

        Args:
            prompt: The user-facing prompt / message content.
            system_prompt: Optional system-level instruction injected before the
                user message (supported by both OpenAI and Anthropic).

        Returns:
            The text content of the LLM's response.

        Raises:
            LLMError: If all retry attempts are exhausted or a non-retryable
                error occurs.
        """
        attempt = 0
        last_exc: Optional[Exception] = None

        while attempt <= self.max_retries:
            try:
                if self.provider == LLMProvider.OPENAI:
                    return self._call_openai(prompt, system_prompt)
                return self._call_anthropic(prompt, system_prompt)
            except (openai.RateLimitError, openai.APIStatusError) as exc:
                last_exc = exc
                if not self._is_retryable_openai(exc):
                    raise LLMError(f"Non-retryable OpenAI error: {exc}") from exc
            except anthropic.RateLimitError as exc:
                last_exc = exc
            except anthropic.APIStatusError as exc:
                last_exc = exc
                if not self._is_retryable_anthropic(exc):
                    raise LLMError(f"Non-retryable Anthropic error: {exc}") from exc
            except (openai.APIConnectionError, anthropic.APIConnectionError) as exc:
                last_exc = exc
            except LLMError:
                raise
            except Exception as exc:
                raise LLMError(f"Unexpected error calling LLM: {exc}") from exc

            delay = _RETRY_BASE_DELAY * (_RETRY_BACKOFF ** attempt)
            logger.warning(
                "LLM API error (attempt %d/%d): %s.  Retrying in %.1fs …",
                attempt + 1,
                self.max_retries,
                last_exc,
                delay,
            )
            time.sleep(delay)
            attempt += 1

        raise LLMError(
            f"LLM call failed after {self.max_retries} retries: {last_exc}"
        )

    def render_and_complete(
        self,
        template_name: str,
        system_prompt: Optional[str] = None,
        **context: Any,
    ) -> str:
        """Render a Jinja2 prompt template and send it to the LLM.

        This is a convenience wrapper around :meth:`render` and
        :meth:`complete`.

        Args:
            template_name: Filename of the Jinja2 template (e.g.
                ``triage_prompt.j2``).
            system_prompt: Optional system-level instruction.
            **context: Variables passed to the template renderer.

        Returns:
            The text content of the LLM's response.

        Raises:
            PromptTemplateError: If the template fails to render.
            LLMError: If the API call fails.
        """
        prompt = self.renderer.render(template_name, **context)
        logger.debug(
            "Sending prompt (template=%s, len=%d chars) to %s/%s.",
            template_name,
            len(prompt),
            self.provider,
            self.model,
        )
        return self.complete(prompt, system_prompt=system_prompt)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_openai(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Execute a chat-completion call against the OpenAI API."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._openai_client.chat.completions.create(  # type: ignore[union-attr]
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            raise LLMError("OpenAI returned an empty response content.")
        return content

    def _call_anthropic(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Execute a messages call against the Anthropic API."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._anthropic_client.messages.create(**kwargs)  # type: ignore[union-attr]
        if not response.content:
            raise LLMError("Anthropic returned an empty response content.")
        # Content is a list of content blocks; extract text from the first one.
        block = response.content[0]
        if hasattr(block, "text"):
            return block.text
        raise LLMError(f"Unexpected Anthropic content block type: {type(block)}")

    @staticmethod
    def _is_retryable_openai(exc: openai.APIStatusError) -> bool:
        """Return True if the OpenAI status error is transient and worth retrying."""
        retryable_codes = {429, 500, 502, 503, 504}
        return getattr(exc, "status_code", None) in retryable_codes

    @staticmethod
    def _is_retryable_anthropic(exc: anthropic.APIStatusError) -> bool:
        """Return True if the Anthropic status error is transient and worth retrying."""
        retryable_codes = {429, 500, 502, 503, 504}
        return getattr(exc, "status_code", None) in retryable_codes
