from pettingzoo.atari import warlords_v3

class WarlordsEnv:
    def __init__(self, render_mode="human"):
        self.env = warlords_v3.env(render_mode=render_mode, full_action_space=False)
        self.env.reset()

    def reset(self):
        self.env.reset()
        return self.env.observe(self.env.agent_selection)

    def step(self, action):
        self.env.step(action)
        obs    = self.env.observe(self.env.agent_selection)
        reward = self.env.rewards[self.env.agent_selection]
        done   = self.env.terminations[self.env.agent_selection]
        info   = self.env.infos[self.env.agent_selection]
        return obs, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def agent(self):
        """Geef de huidige agent terug (bv. 'first_0')."""
        return self.env.agent_selection

    def action_space(self):
        """ActionSpace van de huidige agent."""
        return self.env.action_space(self.env.agent_selection)
