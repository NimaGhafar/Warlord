import gym

class WarlordsEnv:
    def __init__(self, bins = 10):
        self.env = gym.make("ALE/Warlords-v3", render_mode="human")
        self.bins = bins

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()