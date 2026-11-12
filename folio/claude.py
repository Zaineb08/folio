"""Claude as an optional answer model, through Anthropic's API.

It's opt-in, because choosing it sends the passages found for each question to Anthropic
and costs money per question. The API key is kept in Windows Credential Manager, is never
sent back to the browser, and never goes anywhere except Anthropic.

Security research can trip Claude's safety classifiers, and a decline comes back as a
successful response with stop_reason "refusal". So every answer checks for that, and
requests opt in to Anthropic's server-side fallback, which retries on another model.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Iterator

from .errors import Declined, LLMError

log = logging.getLogger("folio.claude")

MODELS = {"claude-opus-5-5": "Claude Opus 5.5"}
_LABELS = {**MODELS, "claude-opus-5": "Claude Opus 5", "claude-opus-4-8": "Claude Opus 4.8"}  # Includes fallback targets.
_PRICES = {  # US dollars per million tokens: input, output.
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_ANSWER_EFFORT = "high"  # Accuracy matters more than cost in thesis work. The model's own default is medium.
_ANSWER_MAX_TOKENS = 16000  # Thinking counts toward this too.
_HELPER_EFFORT = "low"
_HELPER_MAX_TOKENS = 2000
_KEYRING_SERVICE = "Folio"
_KEYRING_USER = "anthropic-api-key"


def is_claude(model: str | None) -> bool:
    return model in MODELS


def label(model: str) -> str:
    """A readable name. Local models are already known by theirs."""
    return _LABELS.get(model, model)


def public_models() -> list[dict]:
    return [{"id": model_id, "label": name} for model_id, name in MODELS.items()]


# ------------------------------------------------------------------- the API key


def get_key() -> str | None:
    environment = os.environ.get("ANTHROPIC_API_KEY")
    if environment:
        return environment
    try:
        import keyring

        return keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER)
    except Exception as exc:  # Keyring backends raise many different types.
        log.warning("Couldn't read the saved API key: %s", type(exc).__name__)
        return None


def has_key() -> bool:
    return bool(get_key())


def save_key(key: str) -> None:
    """Check a key with Anthropic, which costs nothing, then save it."""
    import anthropic
    import keyring

    key = key.strip()
    try:
        anthropic.Anthropic(api_key=key).models.list(limit=1)
    except anthropic.APIError as exc:
        raise _explain(exc) from exc
    try:
        keyring.set_password(_KEYRING_SERVICE, _KEYRING_USER, key)
    except Exception as exc:
        raise LLMError("The key works, but Windows Credential Manager wouldn't save it.") from exc


def delete_key() -> None:
    import keyring

    try:
        keyring.delete_password(_KEYRING_SERVICE, _KEYRING_USER)
    except keyring.errors.PasswordDeleteError:
        pass  # Nothing was saved.


# -------------------------------------------------------------------- answering


def _explain(exc: Exception) -> LLMError:
    """Turn an API error into something the reader can act on. Most specific first."""
    import anthropic

    detail = str(getattr(exc, "message", exc))
    if isinstance(exc, anthropic.AuthenticationError):
        return LLMError("Anthropic didn't accept that API key. Check it, or create a new one at console.anthropic.com.")
    if isinstance(exc, anthropic.PermissionDeniedError):
        return LLMError("This API key doesn't have permission to use that model.")
    if isinstance(exc, anthropic.NotFoundError):
        return LLMError("That Claude model isn't available to this API key.")
    if isinstance(exc, anthropic.RateLimitError):
        return LLMError("Anthropic's rate limit was reached. Wait a minute and try again.")
    if isinstance(exc, anthropic.BadRequestError):
        if "credit balance" in detail.lower():
            return LLMError("Your Anthropic account is out of credit. Add some at console.anthropic.com.")
        return LLMError(f"Anthropic rejected the request: {detail[:300]}")
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMError("Claude took too long to respond. Try again.")
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMError("Couldn't reach Anthropic. Check your internet connection and try again.")
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
        return LLMError("Anthropic's service is having problems right now. Try again shortly.")
    return LLMError(f"Anthropic returned an error: {detail[:300]}")


def _price(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model not in _PRICES:
        return None
    input_rate, output_rate = _PRICES[model]
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def _summarise(final, requested: str) -> dict:
    """What ran and roughly what it cost. usage.iterations lists every attempt when a fallback ran."""
    usage = final.usage
    served_by = final.model
    iterations = getattr(usage, "iterations", None) or []
    fell_back = served_by != requested or any(getattr(entry, "type", "") == "fallback_message" for entry in iterations)
    attempts = [
        (
            getattr(entry, "model", None) or (served_by if getattr(entry, "type", "") == "fallback_message" else requested),
            entry.input_tokens,
            entry.output_tokens,
        )
        for entry in iterations
    ] or [(served_by, usage.input_tokens, usage.output_tokens)]
    prices = [_price(*attempt) for attempt in attempts]
    return {
        "provider": "claude",
        "model": requested,
        "label": label(requested),
        "served_by": served_by,
        "served_label": label(served_by),
        "fell_back": fell_back,
        "input_tokens": sum(attempt[1] for attempt in attempts),
        "output_tokens": sum(attempt[2] for attempt in attempts),
        "cost_usd": None if None in prices else round(sum(prices), 4),
        "truncated": final.stop_reason == "max_tokens",
    }


def chat_stream(
    model: str,
    messages: list[dict],
    effort: str = _ANSWER_EFFORT,
    max_tokens: int = _ANSWER_MAX_TOKENS,
    on_finish: Callable[[dict], None] | None = None,
) -> Iterator[str]:
    """Yield the answer's text as it's generated. Raises Declined if a safety filter refuses."""
    import anthropic

    key = get_key()
    if not key:
        raise LLMError("No Anthropic API key is saved. Add one from the model menu.")
    system = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    conversation = [message for message in messages if message["role"] != "system"]
    options = {"system": system} if system else {}
    try:
        with anthropic.Anthropic(api_key=key).beta.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=conversation,
            output_config={"effort": effort},
            betas=[_FALLBACK_BETA],
            fallbacks="default",
            **options,
        ) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
    except anthropic.APIError as exc:
        raise _explain(exc) from exc
    if final.stop_reason == "refusal":
        raise Declined(getattr(getattr(final, "stop_details", None), "category", None))
    if on_finish:
        on_finish(_summarise(final, model))


def chat_once(model: str, messages: list[dict]) -> str:
    """A small side job, used only when no local model is available to do it for free."""
    return "".join(chat_stream(model, messages, effort=_HELPER_EFFORT, max_tokens=_HELPER_MAX_TOKENS)).strip()
