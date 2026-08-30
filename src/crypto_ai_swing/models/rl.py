from __future__ import annotations

from pathlib import Path


def load_rl_challenger(path: Path, algorithm: str = "MaskablePPO"):
    """Lazy-load an RL challenger. RL is advisory by default."""
    if algorithm == "MaskablePPO":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(str(path))
    if algorithm == "RecurrentPPO":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(str(path))
    raise ValueError(f"Unsupported RL algorithm: {algorithm}")


def rl_has_live_authority() -> bool:
    return False
