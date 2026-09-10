from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import yaml

from crypto_ai_swing.bridge.environment import hydrate_canonical_environment

try:
    from dotenv import dotenv_values
except Exception:
    dotenv_values = None


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _load_swing_environment(root: Path) -> None:
    """Load only orchestrator-owned environment variables from swing `.env`."""

    if dotenv_values is None:
        return
    path = root / ".env"
    if not path.is_file():
        return
    for key, value in dotenv_values(path).items():
        name = str(key)
        if value is None or not str(value).strip():
            continue
        if name == "CRYPTO_REPO_PATH" or name.startswith("CRYPTO_SWING_"):
            os.environ.setdefault(name, str(value))


@dataclass(frozen=True)
class Settings:
    project_root: Path
    crypto_repo_root: Path
    swing: dict
    risk: dict
    universe: dict
    models: dict
    research: dict
    execution: dict
    integrations: dict
    data_quality: dict
    compliance: dict
    nlp: dict
    proactive: dict
    autonomy: dict
    agents: dict
    supervisor: dict

    @classmethod
    def load(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or Path.cwd()).resolve()
        _load_swing_environment(root)
        crypto_root = Path(os.getenv("CRYPTO_REPO_PATH", "../crypto")).expanduser()
        if not crypto_root.is_absolute():
            crypto_root = (root / crypto_root).resolve()

        # Sjagil/crypto owns provider, exchange, risk and live authority env.
        # This happens after the swing-only load so stale duplicated keys in
        # the orchestrator cannot weaken or redirect the canonical engine.
        hydrate_canonical_environment(crypto_root)

        cfg = root / "config"
        return cls(
            project_root=root,
            crypto_repo_root=crypto_root,
            swing=load_yaml(cfg / "swing.yaml"),
            risk=load_yaml(cfg / "risk.yaml"),
            universe=load_yaml(cfg / "universe.yaml"),
            models=load_yaml(cfg / "models.yaml"),
            research=load_yaml(cfg / "research.yaml"),
            execution=load_yaml(cfg / "execution.yaml"),
            integrations=load_yaml(cfg / "integrations.yaml"),
            data_quality=load_yaml(cfg / "data_quality.yaml"),
            compliance=load_yaml(cfg / "compliance.yaml"),
            nlp=load_yaml(cfg / "nlp.yaml"),
            proactive=load_yaml(cfg / "proactive.yaml"),
            autonomy=load_yaml(cfg / "autonomy.yaml"),
            agents=load_yaml(cfg / "agents.yaml"),
            supervisor=load_yaml(cfg / "supervisor.yaml"),
        )
