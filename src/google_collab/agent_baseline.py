import numpy as np

class BaselineAgent:
    """Een simpele rule-based agent voor Warlords."""
    def __init__(self, agent_id=0):
        self.agent_id = agent_id
        self.action_space_size = 3 

    def find_ball(self, observation):
        """Zoekt naar het witte object (de bal) in de observatie."""
        ball_color = [236, 236, 236]
        indices = np.where(np.all(observation == ball_color, axis=-1))
        if len(indices[0]) > 0:
            return np.mean(indices[1]), np.mean(indices[0]) # x, y
        return None, None

    def act(self, observation):
        """Kies een actie op basis van de balpositie."""
        ball_x, _ = self.find_ball(observation)

        if ball_x is None:
            return 0

        screen_center_x = 210 / 2
        
        if ball_x < screen_center_x - 10:
            return 1
        elif ball_x > screen_center_x + 10:
            return 2
        else:
            return 0