"""A2A (Agent-to-Agent) communication protocol.

Implements synchronous, asynchronous, and streaming agent-to-agent messaging
over HTTP using the A2A JSON-RPC protocol.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from enum import Enum
from typing import Any, AsyncIterator, TYPE_CHECKING

import httpx
from pydantic import BaseModel

from agentic.exceptions import A2AError

if TYPE_CHECKING:
    from agentic.a2a.agent_card import AgentCard


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class A2AMessage(BaseModel):
    """A single message in an A2A task."""

    role: str  # "user" or "agent"
    content: str
    metadata: dict[str, Any] = {}


class A2ATask(BaseModel):
    """An A2A task — the unit of work between agents."""

    id: str = ""
    state: TaskState = TaskState.SUBMITTED
    messages: list[A2AMessage] = []
    artifacts: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}

    def latest_message(self) -> A2AMessage | None:
        return self.messages[-1] if self.messages else None

    def result(self) -> str:
        last = self.latest_message()
        return last.content if last else ""


class A2AClient:
    """Client for sending tasks to remote agents via the A2A protocol.

    Supports:
    - Synchronous task send (send_task)
    - Asynchronous polling (send_task_async / get_task)
    - Streaming responses (stream_task)
    """

    def __init__(
        self,
        agent_card: "AgentCard",
        timeout: float = 60.0,
    ) -> None:
        self.agent_card = agent_card
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.agent_card.auth_type == "bearer":
            headers["Authorization"] = f"Bearer {self.agent_card.auth_header}"
        elif self.agent_card.auth_type == "api_key":
            headers["X-API-Key"] = self.agent_card.auth_header
        return headers

    async def send_task(
        self,
        message: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> A2ATask:
        """Send a task synchronously and wait for completion."""
        task_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": task_id,
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "sessionId": session_id or str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}],
                },
                "metadata": metadata or {},
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    self.agent_card.url,
                    json=payload,
                    headers=self._headers(),
                )
                resp.raise_for_status()
                data = resp.json()

            if "error" in data:
                raise A2AError(f"A2A error: {data['error'].get('message')}")

            result = data.get("result", {})
            return A2ATask(
                id=result.get("id", task_id),
                state=TaskState(result.get("status", {}).get("state", "completed")),
                messages=[
                    A2AMessage(role="agent", content=part.get("text", ""))
                    for part in result.get("status", {}).get("message", {}).get("parts", [])
                    if part.get("type") == "text"
                ],
            )
        except A2AError:
            raise
        except Exception as exc:
            raise A2AError(f"A2A communication failed: {exc}") from exc

    async def stream_task(
        self,
        message: str,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream agent responses token-by-token using SSE."""
        task_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": task_id,
            "method": "tasks/sendSubscribe",
            "params": {
                "id": task_id,
                "sessionId": session_id or str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}],
                },
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                self.agent_card.url,
                json=payload,
                headers={**self._headers(), "Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            event = json.loads(data_str)
                            for part in (
                                event.get("result", {})
                                .get("status", {})
                                .get("message", {})
                                .get("parts", [])
                            ):
                                if part.get("type") == "text":
                                    yield part["text"]
                        except json.JSONDecodeError:
                            pass


class A2AServer:
    """Simple A2A server that wraps an agentic BaseAgent.

    Exposes the agent as an HTTP endpoint compatible with the A2A protocol.
    Requires an ASGI framework (e.g., FastAPI) to serve; this class provides
    the core request handling logic.
    """

    def __init__(self, agent: Any, card: "AgentCard") -> None:
        self.agent = agent
        self.card = card

    async def handle_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        method = request_data.get("method", "")
        req_id = request_data.get("id")

        if method == "tasks/send":
            params = request_data.get("params", {})
            task_id = params.get("id", str(uuid.uuid4()))
            message_parts = params.get("message", {}).get("parts", [])
            user_text = " ".join(
                p.get("text", "") for p in message_parts if p.get("type") == "text"
            )

            try:
                result = await self.agent.run(user_text)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "id": task_id,
                        "status": {
                            "state": "completed",
                            "message": {
                                "role": "agent",
                                "parts": [{"type": "text", "text": result.content}],
                            },
                        },
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "id": task_id,
                        "status": {
                            "state": "failed",
                            "message": {"role": "agent", "parts": [{"type": "text", "text": str(exc)}]},
                        },
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }
