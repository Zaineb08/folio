"""Errors from answer models, worded to be shown to the reader as they are."""

from __future__ import annotations


class LLMError(RuntimeError):
    pass


class Declined(LLMError):
    """A safety filter refused the request. Any partial answer has to be discarded."""

    def __init__(self, category: str | None = None) -> None:
        self.category = category
        kind = f" ({category})" if category else ""
        super().__init__(
            f"Claude's safety filter declined this question{kind}. That can happen with security topics, "
            "even for legitimate research. Try rewording it, or switch to the model on this PC."
        )
