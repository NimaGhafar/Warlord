"""
Eenvoudige baseline: kiest elke stap een willekeurige actie.
"""

import numpy as np


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, obs):
        # `obs` wordt hier genegeerd – volledig random policy
        return self.action_space.sample()