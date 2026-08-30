from pathlib import Path
import yaml


def test_crypto_relevant_references_enabled():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config/references.yaml").read_text(encoding="utf-8"))
    refs = cfg["references"]
    for required in (
        "nautilus_trader",
        "vectorbt",
        "optuna",
        "skfolio",
        "stable_baselines3_contrib",
        "kronos",
        "finrl_trading",
    ):
        assert refs[required]["enabled"] is True
    assert refs["vnpy_ib"]["enabled"] is False
