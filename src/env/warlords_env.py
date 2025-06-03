"""
Wrapper om de Atari-omgeving Warlords te bouwen met standaard-preprocessing.
"""

import gym
from gym.wrappers import AtariPreprocessing


def make_env(seed: int | None = None):
    """
    Maakt een Atari Warlords-omgeving met grijswaarden en geschaalde frames.

    Parameters
    ----------
    seed : int | None
        Vast seed getal voor reproduceerbaarheid.  None → geen seed.

    Returns
    -------
    gym.Env
    """
    env = gym.make("ALE/Warlords-v5")

    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)

    if seed is not None:
        env.seed(seed)

    return env