from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import yaml

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


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
        if load_dotenv is not None:
            load_dotenv(root / ".env", override=False)
        crypto_root = Path(os.getenv("CRYPTO_REPO_PATH", "../crypto")).expanduser()
        if not crypto_root.is_absolute():
            crypto_root = (root / crypto_root).resolve()
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
