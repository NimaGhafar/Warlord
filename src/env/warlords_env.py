"""
Atari Warlords-omgeving met preprocessing (frame-skip 4, grijswaarden, schaal 0-1)
– Compatibel met Python 3.8 (Optional-syntax)
– Registreert de ALE-envs door import van `gymnasium_atari`
"""

from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

# ── Zorg dat alle Atari-envs in het registry komen ────────────────────────
try:
    import gymnasium_atari  # noqa: F401  (alleen side-effect: registratie)
except ImportError:
    # Fallback: probeer gewone ale_py — werkt als plugin al eerder geladen is
    import ale_py  # noqa: F401
# ───────────────────────────────────────────────────────────────────────────


def make_env(seed: Optional[int] = None):
    """
    Parameters
    ----------
    seed : Optional[int]
        Vast getal voor reproduceerbaarheid (None → geen seed).

    Returns
    -------
    gymnasium.Env
    """
    env = gym.make("ALE/Warlords-v5")

    # Standaard Atari-preprocessing
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=True,
    )

    # Seed (Gymnasium ≥0.29 gebruikt reset(seed=…))
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:           # oudere versies
            env.seed(seed)

    return env