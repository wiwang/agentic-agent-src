"""Plugin manager — discovers, loads, and manages agentic plugins."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import warnings
from pathlib import Path
from typing import Any, TYPE_CHECKING

from agentic.plugins.base import AgentPlugin, PluginMetadata

if TYPE_CHECKING:
    from agentic.core.agent import BaseAgent


class PluginManager:
    """Manages plugin lifecycle: discovery, loading, and integration.

    Plugins are discovered from:
    1. setuptools ``entry_points`` (group: ``agentic.plugins``)
    2. A directory specified via ``load_from_directory()``
    3. Direct registration via ``register()``

    Usage::

        manager = PluginManager()
        manager.load_from_entry_points()
        manager.load_from_directory("./my_plugins")

        # Apply all plugins to an agent
        manager.apply_to_agent(my_agent)
    """

    def __init__(self) -> None:
        self._plugins: dict[str, AgentPlugin] = {}

    def register(self, plugin: AgentPlugin) -> "PluginManager":
        """Register a plugin instance directly."""
        name = plugin.metadata.name
        plugin.on_load()
        self._plugins[name] = plugin
        return self

    def unregister(self, name: str) -> None:
        plugin = self._plugins.pop(name, None)
        if plugin:
            plugin.on_unload()

    def get(self, name: str) -> AgentPlugin | None:
        return self._plugins.get(name)

    def all_plugins(self) -> list[AgentPlugin]:
        return list(self._plugins.values())

    def load_from_entry_points(self, group: str = "agentic.plugins") -> int:
        """Discover and load plugins via setuptools entry_points.

        Returns the number of plugins successfully loaded.
        """
        try:
            from importlib.metadata import entry_points
        except ImportError:
            try:
                from importlib_metadata import entry_points  # type: ignore
            except ImportError:
                return 0

        loaded = 0
        eps = entry_points(group=group)
        for ep in eps:
            try:
                obj = ep.load()
                # Instantiate if it's a class
                instance = obj() if inspect.isclass(obj) else obj
                if isinstance(instance, AgentPlugin):
                    self.register(instance)
                    loaded += 1
                else:
                    warnings.warn(
                        f"Entry point '{ep.name}' did not yield an AgentPlugin instance."
                    )
            except Exception as exc:
                warnings.warn(f"Failed to load plugin '{ep.name}': {exc}")

        return loaded

    def load_from_directory(self, directory: str) -> int:
        """Scan a directory for Python modules containing AgentPlugin subclasses.

        Returns the number of plugins successfully loaded.
        """
        path = Path(directory)
        if not path.exists():
            return 0

        loaded = 0
        for py_file in sorted(path.glob("*.py")):
            if py_file.stem.startswith("_"):
                continue
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)  # type: ignore[arg-type]
            except Exception as exc:
                warnings.warn(f"Failed to import plugin module '{py_file}': {exc}")
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, AgentPlugin)
                    and obj is not AgentPlugin
                    and not inspect.isabstract(obj)
                ):
                    try:
                        instance = obj()
                        self.register(instance)
                        loaded += 1
                    except Exception as exc:
                        warnings.warn(f"Failed to instantiate plugin '{obj.__name__}': {exc}")

        return loaded

    def load_module(self, module_path: str) -> int:
        """Load plugins from a dotted Python module path, e.g., 'mypackage.plugins'."""
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            warnings.warn(f"Failed to import plugin module '{module_path}': {exc}")
            return 0

        loaded = 0
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, AgentPlugin)
                and obj is not AgentPlugin
                and not inspect.isabstract(obj)
            ):
                try:
                    self.register(obj())
                    loaded += 1
                except Exception as exc:
                    warnings.warn(f"Failed to load plugin '{obj.__name__}': {exc}")
        return loaded

    def apply_to_agent(self, agent: "BaseAgent") -> None:
        """Register all plugin tools and guardrails with an agent."""
        for plugin in self._plugins.values():
            for tool in plugin.get_tools():
                agent.add_tool(tool)
            for guard in plugin.get_guardrails():
                agent.add_guardrail(guard)

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "name": p.metadata.name,
                "version": p.metadata.version,
                "description": p.metadata.description,
                "tools": [t.name for t in p.get_tools()],
            }
            for p in self._plugins.values()
        ]


# Global default plugin manager
_default_manager = PluginManager()


def get_plugin_manager() -> PluginManager:
    return _default_manager
