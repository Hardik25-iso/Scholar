"""The LLM seam: does swapping providers stay a config change?

These assert the contract the seam promises — that generation behaves the same
whichever provider is configured — without calling a real model. A live call
would need a key, cost money, and fail in CI for reasons unrelated to the code.
"""
import pytest

from backend import generator, llm
from backend.config import settings

GROQ = {
    "llm_base_url": "https://api.groq.com/openai/v1",
    "llm_api_key": "gsk_test_not_a_real_key",
    "llm_model": "llama-3.3-70b-versatile",
}


@pytest.fixture(autouse=True)
def _restore_provider():
    """Every test gets a clean provider; none leaks its config into the next."""
    original = {k: getattr(settings, k) for k in
                ("llm_provider", "llm_base_url", "llm_api_key", "llm_model")}
    llm.reset_provider()
    yield
    for k, v in original.items():
        setattr(settings, k, v)
    llm.reset_provider()


def _use(provider: str, **overrides):
    settings.llm_provider = provider
    for k, v in overrides.items():
        setattr(settings, k, v)
    llm.reset_provider()


def test_default_is_the_local_model():
    """Nothing configured means Ollama — a fresh clone must not need a key."""
    _use("ollama")
    assert isinstance(llm.get_provider(), llm.OllamaProvider)


def test_hosted_needs_its_config_and_says_which_parts_are_missing():
    """A missing key must fail at startup, not at the first user's question —
    and the error must name what to set, since that is the whole fix."""
    _use("hosted", llm_base_url="", llm_api_key="", llm_model="")
    with pytest.raises(RuntimeError) as exc:
        llm.get_provider()
    message = str(exc.value)
    for setting in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        assert setting in message


def test_an_unknown_provider_is_rejected_by_name():
    """A typo in LLM_PROVIDER should be a loud error, not a silent fallback to
    a model the operator did not choose."""
    _use("gpt")
    with pytest.raises(RuntimeError, match="unknown LLM_PROVIDER"):
        llm.get_provider()


def test_switching_vendor_is_two_settings_not_a_new_class():
    """The point of the seam: Groq and Gemini are the same code path."""
    _use("hosted", **GROQ)
    groq = llm.get_provider()
    assert groq.MODEL == "llama-3.3-70b-versatile"

    _use("hosted",
         llm_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
         llm_model="gemini-2.0-flash")
    gemini = llm.get_provider()

    assert type(groq) is type(gemini) is llm.HostedProvider
    assert gemini.MODEL == "gemini-2.0-flash"
    assert "googleapis" in str(gemini._client.base_url)


def test_every_provider_bounds_its_requests():
    """An unbounded request pins a worker forever — the failure that takes the
    whole app down quietly. Both providers must carry the configured timeout."""
    _use("hosted", **GROQ)
    assert llm.get_provider()._client.timeout == settings.llm_timeout_seconds

    _use("ollama")
    assert llm.get_provider()._client._client.timeout is not None


def test_generation_is_reproducible_on_both_providers():
    """The audit log promises the same question gives the same answer, which
    only holds if sampling is pinned wherever generation runs."""
    _use("ollama")
    assert generator.active_temperature() == 0.0

    _use("hosted", **GROQ)
    assert generator.active_temperature() == 0.0


def test_the_audit_log_records_which_model_actually_answered():
    """"Which model wrote this" is the field that explains why two answers to
    the same question differ, so it must follow the live provider."""
    _use("ollama")
    assert generator.active_model() == "gemma3:4b"

    _use("hosted", **GROQ)
    assert generator.active_model() == "llama-3.3-70b-versatile"


def test_provider_failures_surface_as_one_error_type():
    """api.py turns LLMUnavailable into a clean 503. A provider that leaked its
    own SDK's exception would surface as a 500 with a stack trace instead."""
    _use("hosted", **GROQ)
    provider = llm.get_provider()
    import openai

    for raised, expected in [
        (openai.AuthenticationError, "LLM_API_KEY"),
        (openai.RateLimitError, "rate limit"),
        (openai.NotFoundError, "does not serve a model"),
    ]:
        wrapped = provider._wrap(raised.__new__(raised))
        assert isinstance(wrapped, llm.LLMUnavailable)
        assert expected in str(wrapped)
