"""ToolRegistry — central store for tool discovery and plugin-based loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from agentic.exceptions import ToolNotFoundError
from agentic.llm.base import ToolSchema

if TYPE_CHECKING:
    from agentic.tools.base import BaseTool


class ToolRegistry:
    """Registry that maps tool names to BaseTool instances.

    Supports:
    - Manual registration via ``register()``
    - Plugin-based discovery via ``load_from_entry_points()``
    - Namespace filtering
    """

    def __init__(self) -> None:
        self._tools: dict[str, "BaseTool"] = {}

    def register(self, tool: "BaseTool") -> None:
        """Register a tool under its name."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> "BaseTool":
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def get_or_none(self, name: str) -> "BaseTool | None":
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def all_tools(self) -> list["BaseTool"]:
        return list(self._tools.values())

    def schemas(self) -> list[ToolSchema]:
        return [t.to_schema() for t in self._tools.values()]

    def __iter__(self) -> Iterator["BaseTool"]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def load_from_entry_points(self, group: str = "agentic.plugins") -> None:
        """Discover and register tools exposed via setuptools entry_points."""
        try:
            from importlib.metadata import entry_points
        except ImportError:
            from importlib_metadata import entry_points  # type: ignore[no-redef]

        eps = entry_points(group=group)
        for ep in eps:
            try:
                obj = ep.load()
                if callable(obj):
                    instance = obj()
                else:
                    instance = obj
                from agentic.tools.base import BaseTool
                if isinstance(instance, BaseTool):
                    self.register(instance)
            except Exception as exc:
                import warnings
                warnings.warn(f"Failed to load tool from entry point '{ep.name}': {exc}")

    def load_from_directory(self, directory: str) -> None:
        """Scan a directory for Python modules and register any BaseTool subclasses."""
        import importlib.util
        import inspect
        from pathlib import Path
        from agentic.tools.base import BaseTool

        path = Path(directory)
        for py_file in path.glob("*.py"):
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)  # type: ignore[arg-type]
            except Exception:
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseTool)
                    and obj is not BaseTool
                    and not inspect.isabstract(obj)
                ):
                    try:
                        self.register(obj())
                    except Exception:
                        pass


# Global default registry
_default_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    return _default_registry
