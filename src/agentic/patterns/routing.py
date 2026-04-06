"""Routing pattern — Chapter 2.

Routes input to the most appropriate handler (agent/pattern/function)
using either LLM-based classification or rule-based matching.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from agentic.core.message import Message
from agentic.exceptions import RoutingError
from agentic.patterns.base import BasePattern, PatternResult

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from agentic.llm.base import BaseLLMProvider


RouteHandler = Callable[[str, Any], Awaitable[str]]


class Route(BaseModel):
    """A named route with an optional keyword pattern for rule-based matching."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    keywords: list[str] = []  # For rule-based matching


ROUTER_PROMPT = """Classify the following user input into exactly one category.
Respond with ONLY the category name, nothing else.

Categories:
{categories}

User input: {input}

Category:"""


class LLMRouter(BasePattern):
    """Routes input to a handler using LLM-based intent classification."""

    def __init__(
        self,
        llm: "BaseLLMProvider",
        routes: list[Route] | None = None,
        handlers: dict[str, RouteHandler] | None = None,
        default_route: str | None = None,
    ) -> None:
        self.llm = llm
        self.routes: list[Route] = routes or []
        self.handlers: dict[str, RouteHandler] = handlers or {}
        self.default_route = default_route

    def add_route(
        self,
        name: str,
        description: str,
        handler: RouteHandler,
        keywords: list[str] | None = None,
    ) -> "LLMRouter":
        self.routes.append(Route(name=name, description=description, keywords=keywords or []))
        self.handlers[name] = handler
        return self

    async def classify(self, input: str) -> str:
        """Ask the LLM to classify the input into a route name."""
        categories = "\n".join(
            f"- {r.name}: {r.description}" for r in self.routes
        )
        prompt = ROUTER_PROMPT.format(categories=categories, input=input)
        response = await self.llm.generate([Message.human(prompt)])
        chosen = response.message.content.strip().lower()
        # Match to a known route name
        for route in self.routes:
            if route.name.lower() == chosen or chosen in route.name.lower():
                return route.name
        return self.default_route or (self.routes[0].name if self.routes else "")

    async def run(
        self,
        input: str,
        context: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> PatternResult:
        if not self.routes:
            raise RoutingError("No routes configured.")

        route_name = await self.classify(input)
        handler = self.handlers.get(route_name)

        if handler is None:
            if self.default_route and self.default_route in self.handlers:
                handler = self.handlers[self.default_route]
                route_name = self.default_route
            else:
                raise RoutingError(f"No handler for route '{route_name}'.")

        output = await handler(input, context)
        return PatternResult(
            output=output,
            steps=[f"Routed to: {route_name}"],
            metadata={"route": route_name},
        )


class RuleBasedRouter(BasePattern):
    """Routes input using keyword/regex matching — no LLM call required."""

    def __init__(
        self,
        routes: list[Route] | None = None,
        handlers: dict[str, RouteHandler] | None = None,
        default_route: str | None = None,
    ) -> None:
        self.routes: list[Route] = routes or []
        self.handlers: dict[str, RouteHandler] = handlers or {}
        self.default_route = default_route

    def add_route(
        self,
        name: str,
        description: str,
        handler: RouteHandler,
        keywords: list[str],
    ) -> "RuleBasedRouter":
        self.routes.append(Route(name=name, description=description, keywords=keywords))
        self.handlers[name] = handler
        return self

    def _match(self, input: str) -> str | None:
        lower = input.lower()
        for route in self.routes:
            for kw in route.keywords:
                if re.search(kw, lower):
                    return route.name
        return self.default_route

    async def run(
        self,
        input: str,
        context: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> PatternResult:
        route_name = self._match(input)
        if route_name is None:
            raise RoutingError("No matching route found and no default configured.")

        handler = self.handlers.get(route_name)
        if handler is None:
            raise RoutingError(f"No handler for route '{route_name}'.")

        output = await handler(input, context)
        return PatternResult(
            output=output,
            steps=[f"Matched route: {route_name}"],
            metadata={"route": route_name, "method": "rule-based"},
        )
