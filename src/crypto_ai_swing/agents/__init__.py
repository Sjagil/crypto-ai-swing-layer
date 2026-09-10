from __future__ import annotations

from importlib import import_module

__all__ = [
    "AgentManager",
    "AgentRuntime",
    "AgentTrainer",
    "ChiefAgent",
    "PerformanceGovernor",
    "TrainingResult",
    "build_research_agent_panel",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentManager": (".manager", "AgentManager"),
    "AgentRuntime": (".runtime", "AgentRuntime"),
    "AgentTrainer": (".training", "AgentTrainer"),
    "ChiefAgent": (".chief", "ChiefAgent"),
    "PerformanceGovernor": (".performance_governor", "PerformanceGovernor"),
    "TrainingResult": (".training", "TrainingResult"),
    "build_research_agent_panel": (".panel", "build_research_agent_panel"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
