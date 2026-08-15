"""The LLM seam: one interface, two providers.

Scholar's grounding contract lives in generator.py and does not change here.
This module only answers "who writes the words":

  ollama — gemma3:4b on this machine. Free, private, no API key. The right
           default for development, and impossible to deploy: a hosted box has
           no Ollama, and shipping a 3.3 GB model to every visitor is not a
           thing. This is why Scholar could be run but not published.
  hosted — any OpenAI-compatible chat endpoint, chosen by base URL. Groq,
           Google's Gemini/Gemma API and OpenRouter all speak this protocol and
           all have free tiers, so one implementation covers every one of them
           and switching vendors is two environment variables, not a code
           change. (It also covers paid vendors and self-hosted servers like
           vLLM or llama.cpp, which speak the same protocol.)

Both implement the same three operations, so retrieval, reranking, citations and
streaming behave identically whichever one is configured. Switching is a config
change (LLM_PROVIDER), not a code change — which is the whole point of a seam.
"""
from collections.abc import Iterator
from typing import Protocol

import httpx

from backend.config import settings


class LLMUnavailable(RuntimeError):
    """The model did not answer in time, or the provider is unreachable.

    Raised instead of letting a provider's own exception escape, so the API
    layer can turn any provider's failure into one clean 503.
    """


class LLMProvider(Protocol):
    """What a generator needs from a model. Deliberately tiny."""

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return a full response. Used for answers and question condensing."""

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str]:
        """Yield incremental text deltas as the model writes."""

    def ping(self) -> None:
        """Cheap liveness probe. Raises LLMUnavailable if the model can't serve."""


# ——————————————————————————— Ollama (local) ———————————————————————————


class OllamaProvider:
    """gemma3:4b via a local Ollama daemon.

    temperature=0 for deterministic, faithful answers — determinism is what the
    audit log's "same question, same answer" promise rests on. The hosted
    provider pins it the same way, so the promise holds on either path.
    """

    MODEL = "gemma3:4b"
    TEMPERATURE = 0.0

    def __init__(self) -> None:
        import ollama

        # Explicit timeout: the module-level ollama.chat helper has none, so a
        # hung model would pin a threadpool worker forever.
        self._client = ollama.Client(timeout=settings.llm_timeout_seconds)

    def _options(self, max_tokens: int) -> dict:
        return {"temperature": self.TEMPERATURE, "num_predict": max_tokens}

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        try:
            response = self._client.chat(
                model=self.MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                options=self._options(max_tokens),
            )
        except (httpx.TimeoutException, httpx.ConnectError, ConnectionError) as exc:
            raise LLMUnavailable(
                f"the local model did not respond within "
                f"{settings.llm_timeout_seconds:.0f}s (is Ollama running?)"
            ) from exc
        return response["message"]["content"].strip()

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str]:
        try:
            stream = self._client.chat(
                model=self.MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                options=self._options(max_tokens),
                stream=True,
            )
            # The timeout covers the gaps BETWEEN chunks too, so a model that
            # stalls mid-answer raises rather than hanging the response open.
            for part in stream:
                delta = part["message"]["content"]
                if delta:
                    yield delta
        except (httpx.TimeoutException, httpx.ConnectError, ConnectionError) as exc:
            raise LLMUnavailable("the local model stalled mid-answer") from exc

    def ping(self) -> None:
        try:
            result = self._client.list()
        except (httpx.TimeoutException, httpx.ConnectError, ConnectionError) as exc:
            raise LLMUnavailable("Ollama is not reachable") from exc
        names = {m.get("model") or m.get("name") for m in result.get("models", [])}
        if self.MODEL not in names:
            raise LLMUnavailable(
                f"model {self.MODEL} is not pulled (run: ollama pull {self.MODEL})"
            )

    def warm(self) -> None:
        """Preload the model so the first real answer doesn't pay the cold load."""
        self._client.chat(
            model=self.MODEL,
            messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 1},
        )


# ———————————————— Hosted, via any OpenAI-compatible endpoint ————————————————


class HostedProvider:
    """Any chat endpoint that speaks the OpenAI protocol, selected by base URL.

    Deliberately vendor-agnostic. The protocol is the de-facto standard, so this
    one class covers every free-tier option worth deploying on:

      Groq      https://api.groq.com/openai/v1          (fast, generous free tier)
      Gemini    https://generativelanguage.googleapis.com/v1beta/openai/
      OpenRouter https://openrouter.ai/api/v1           (models suffixed :free)

    …and paid or self-hosted servers (vLLM, llama.cpp, LM Studio) at no extra
    cost, since they speak it too. Changing vendor is LLM_BASE_URL + LLM_MODEL.

    Temperature IS sent here: unlike some vendors' native APIs, the OpenAI
    protocol accepts it, and 0 is what keeps answers reproducible — which the
    audit log's "same question, same answer" promise depends on.
    """

    TEMPERATURE = 0.0

    def __init__(self) -> None:
        import openai

        missing = [
            name for name, value in (
                ("LLM_BASE_URL", settings.llm_base_url),
                ("LLM_API_KEY", settings.llm_api_key),
                ("LLM_MODEL", settings.llm_model),
            ) if not value
        ]
        if missing:
            raise RuntimeError(
                f"LLM_PROVIDER=hosted but {', '.join(missing)} not set. "
                f"For Groq: LLM_BASE_URL=https://api.groq.com/openai/v1, "
                f"LLM_MODEL=llama-3.3-70b-versatile, LLM_API_KEY=<your key>"
            )
        self._openai = openai
        self.MODEL = settings.llm_model
        self._client = openai.OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_seconds,  # seconds
            max_retries=2,
        )

    def _wrap(self, exc: Exception) -> LLMUnavailable:
        """Normalise a vendor's failure into Scholar's one 503-able error.

        Free tiers rate-limit aggressively, so 429 gets its own message — "try
        again shortly" is actionable, a stack trace is not.
        """
        o = self._openai
        if isinstance(exc, o.RateLimitError):
            return LLMUnavailable(
                "the model provider's rate limit was reached — try again shortly"
            )
        if isinstance(exc, o.AuthenticationError):
            return LLMUnavailable("the configured LLM_API_KEY was rejected")
        if isinstance(exc, o.NotFoundError):
            return LLMUnavailable(
                f"the provider does not serve a model named {self.MODEL!r} "
                f"— check LLM_MODEL against the provider's current model list"
            )
        if isinstance(exc, o.APIConnectionError):
            return LLMUnavailable("could not reach the model provider")
        return LLMUnavailable(f"the model provider returned an error: {exc}")

    def _messages(self, system: str, user: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        o = self._openai
        try:
            completion = self._client.chat.completions.create(
                model=self.MODEL,
                messages=self._messages(system, user),
                max_tokens=max_tokens,
                temperature=self.TEMPERATURE,
            )
        except (o.APIStatusError, o.APIConnectionError) as exc:
            raise self._wrap(exc) from exc
        # content is nullable in the protocol (e.g. a filtered response), so a
        # bare .strip() would crash on exactly the responses worth handling.
        return (completion.choices[0].message.content or "").strip()

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str]:
        o = self._openai
        try:
            stream = self._client.chat.completions.create(
                model=self.MODEL,
                messages=self._messages(system, user),
                max_tokens=max_tokens,
                temperature=self.TEMPERATURE,
                stream=True,
            )
            for chunk in stream:
                # A chunk can carry no choices at all (some vendors send a
                # usage-only final chunk), so index defensively.
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except (o.APIStatusError, o.APIConnectionError) as exc:
            raise self._wrap(exc) from exc

    def ping(self) -> None:
        """Confirm the key works and the configured model is actually served.

        Listing models is cheap and, unlike a generation, does not consume the
        free tier's token budget every time a load balancer probes /health.
        """
        o = self._openai
        try:
            served = {m.id for m in self._client.models.list()}
        except (o.APIStatusError, o.APIConnectionError) as exc:
            raise self._wrap(exc) from exc
        # Some gateways (notably OpenRouter) list thousands of models or none at
        # all; only fail when the listing is present AND excludes our model.
        if served and self.MODEL not in served:
            raise LLMUnavailable(
                f"the provider does not serve a model named {self.MODEL!r} "
                f"— check LLM_MODEL against the provider's current model list"
            )

    def warm(self) -> None:
        """Nothing to warm: a hosted model has no local load to pay for."""


# ——————————————————————————————— selection ———————————————————————————————

_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """The configured provider, built once and reused."""
    global _provider
    if _provider is None:
        if settings.llm_provider == "hosted":
            _provider = HostedProvider()
        elif settings.llm_provider == "ollama":
            _provider = OllamaProvider()
        else:
            raise RuntimeError(
                f"unknown LLM_PROVIDER {settings.llm_provider!r} "
                f"(expected 'ollama' or 'hosted')"
            )
    return _provider


def reset_provider() -> None:
    """Drop the cached provider. For tests that switch providers."""
    global _provider
    _provider = None
