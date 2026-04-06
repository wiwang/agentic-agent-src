"""Input guardrails — validate and sanitize user inputs."""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from agentic.exceptions import InputGuardrailError
from agentic.guardrails.base import BaseGuardrail

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


# Common jailbreak patterns
_JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"forget\s+(everything|all)\s+(you\s+)?(know|were\s+told)",
    r"you\s+are\s+now\s+(a\s+)?(?:dan|jailbroken|evil|unrestricted)",
    r"do\s+anything\s+now",
    r"pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(?:evil|unrestricted|unethical)",
    r"bypass\s+(your\s+)?(safety|content|ethical)\s+(filters?|guidelines?|rules?)",
    r"disregard\s+(your\s+)?(ethical|moral|safety)",
    r"roleplay\s+as.*(?:evil|harmful|dangerous)",
    r"act\s+as\s+if\s+you\s+(have\s+no|don't\s+have)\s+(restrictions|guidelines)",
]

_COMPILED_JAILBREAKS = [re.compile(p, re.IGNORECASE) for p in _JAILBREAK_PATTERNS]

# Prompt injection markers
_INJECTION_PATTERNS = [
    r"<\s*system\s*>",
    r"\[INST\]",
    r"###\s*system\s*:",
    r"<<SYS>>",
    r"<\|im_start\|>",
]
_COMPILED_INJECTIONS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


class JailbreakDetector(BaseGuardrail):
    """Detects and blocks common jailbreak and prompt injection attempts."""

    def __init__(self, raise_on_detect: bool = True) -> None:
        self.raise_on_detect = raise_on_detect

    async def check_input(self, text: str, context: "AgentContext | None" = None) -> str:
        # Check jailbreak patterns
        for pattern in _COMPILED_JAILBREAKS:
            if pattern.search(text):
                if self.raise_on_detect:
                    raise InputGuardrailError(
                        f"Input blocked: potential jailbreak attempt detected."
                    )
                return "[Input blocked: potential jailbreak attempt]"

        # Check prompt injection
        for pattern in _COMPILED_INJECTIONS:
            if pattern.search(text):
                if self.raise_on_detect:
                    raise InputGuardrailError(
                        "Input blocked: prompt injection attempt detected."
                    )
                return "[Input blocked: prompt injection attempt]"

        return text


class LengthGuard(BaseGuardrail):
    """Enforces minimum and maximum input length."""

    def __init__(self, min_length: int = 1, max_length: int = 50000) -> None:
        self.min_length = min_length
        self.max_length = max_length

    async def check_input(self, text: str, context: "AgentContext | None" = None) -> str:
        if len(text) < self.min_length:
            raise InputGuardrailError(
                f"Input too short: {len(text)} < {self.min_length} characters."
            )
        if len(text) > self.max_length:
            # Truncate rather than block
            return text[: self.max_length]
        return text


class PIIRedactor(BaseGuardrail):
    """Redacts common PII patterns from inputs and outputs."""

    _PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
        ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
        ("phone", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
        ("ssn", re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"), "[SSN]"),
        ("credit_card", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[CREDIT_CARD]"),
    ]

    def __init__(self, redact_in_output: bool = True, redact_in_input: bool = False) -> None:
        self.redact_in_output = redact_in_output
        self.redact_in_input = redact_in_input

    def _redact(self, text: str) -> str:
        for _, pattern, replacement in self._PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    async def check_input(self, text: str, context: "AgentContext | None" = None) -> str:
        return self._redact(text) if self.redact_in_input else text

    async def check_output(self, text: str, context: "AgentContext | None" = None) -> str:
        return self._redact(text) if self.redact_in_output else text


class ContentPolicyGuard(BaseGuardrail):
    """Blocks inputs containing explicitly forbidden topics/keywords."""

    def __init__(
        self,
        forbidden_keywords: list[str] | None = None,
        case_sensitive: bool = False,
    ) -> None:
        self.case_sensitive = case_sensitive
        raw = forbidden_keywords or []
        if not case_sensitive:
            self._keywords = [k.lower() for k in raw]
        else:
            self._keywords = raw

    async def check_input(self, text: str, context: "AgentContext | None" = None) -> str:
        check_text = text if self.case_sensitive else text.lower()
        for kw in self._keywords:
            if kw in check_text:
                raise InputGuardrailError(
                    f"Input blocked by content policy (matched forbidden keyword)."
                )
        return text
