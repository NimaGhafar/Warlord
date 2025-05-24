import gym
from ale_py import ALEInterface
from gym.wrappers import AtariPreprocessing

def make_env(seed=None):
    env = gym.make('ALE/Warlords-v5')
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)
    if seed is not None:
        env.seed(seed)
    return env