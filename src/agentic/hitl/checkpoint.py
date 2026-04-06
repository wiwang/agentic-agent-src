"""Human-in-the-Loop (HITL) checkpoints.

Implements pause-and-wait semantics so a human can review, approve,
reject, or modify agent state before execution continues.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from pydantic import BaseModel

from agentic.exceptions import HITLError, HITLRejectionError

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


class CheckpointDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    SKIP = "skip"


class CheckpointReview(BaseModel):
    """Human's decision at a checkpoint."""

    decision: CheckpointDecision
    modified_content: str | None = None
    comment: str = ""
    reviewer: str = "human"


ReviewCallback = Callable[[str, str, Any], Awaitable[CheckpointReview]]


async def _console_reviewer(
    checkpoint_name: str, content: str, context: Any
) -> CheckpointReview:
    """Default interactive console reviewer."""
    print(f"\n{'='*60}")
    print(f"HITL Checkpoint: {checkpoint_name}")
    print(f"{'='*60}")
    print(f"Content for review:\n{content}")
    print(f"\nOptions: [a]pprove / [r]eject / [m]odify / [s]kip")
    try:
        choice = input("Decision: ").strip().lower()
    except EOFError:
        choice = "a"

    if choice.startswith("r"):
        comment = input("Rejection reason: ") if not choice.startswith("re") else ""
        return CheckpointReview(decision=CheckpointDecision.REJECT, comment=comment)
    elif choice.startswith("m"):
        print("Enter modified content (end with a line containing only 'END'):")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "END":
                break
            lines.append(line)
        return CheckpointReview(
            decision=CheckpointDecision.MODIFY,
            modified_content="\n".join(lines),
        )
    elif choice.startswith("s"):
        return CheckpointReview(decision=CheckpointDecision.SKIP)
    else:
        return CheckpointReview(decision=CheckpointDecision.APPROVE)


class HITLCheckpoint:
    """Defines a single HITL checkpoint.

    Usage::

        checkpoint = HITLCheckpoint(
            name="approve_plan",
            description="Review the generated plan before execution",
            reviewer=my_async_reviewer,
        )
        content = await checkpoint.review(plan_text, context)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        reviewer: ReviewCallback | None = None,
        auto_approve_after: float | None = None,  # seconds
        required: bool = True,
    ) -> None:
        self.name = name
        self.description = description
        self._reviewer = reviewer or _console_reviewer
        self.auto_approve_after = auto_approve_after
        self.required = required

    async def review(
        self,
        content: str,
        context: "AgentContext | None" = None,
    ) -> str:
        """Present content to human reviewer and return (possibly modified) content.

        Raises:
            HITLRejectionError: if the human rejects the checkpoint.
        """
        if self.auto_approve_after is not None:
            try:
                decision = await asyncio.wait_for(
                    self._reviewer(self.name, content, context),
                    timeout=self.auto_approve_after,
                )
            except asyncio.TimeoutError:
                return content  # Auto-approve on timeout
        else:
            decision = await self._reviewer(self.name, content, context)

        if decision.decision == CheckpointDecision.REJECT:
            raise HITLRejectionError(
                f"Checkpoint '{self.name}' rejected: {decision.comment}"
            )
        elif decision.decision == CheckpointDecision.MODIFY:
            return decision.modified_content or content
        else:
            # APPROVE or SKIP
            return content


class HITLManager:
    """Manages multiple HITL checkpoints for an agent."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, HITLCheckpoint] = {}

    def register(self, checkpoint: HITLCheckpoint) -> "HITLManager":
        self._checkpoints[checkpoint.name] = checkpoint
        return self

    def get(self, name: str) -> HITLCheckpoint | None:
        return self._checkpoints.get(name)

    async def checkpoint(
        self,
        name: str,
        content: str,
        context: "AgentContext | None" = None,
    ) -> str:
        """Run a named checkpoint. Returns (possibly modified) content."""
        cp = self._checkpoints.get(name)
        if cp is None:
            raise HITLError(f"Checkpoint '{name}' not registered.")
        return await cp.review(content, context)
