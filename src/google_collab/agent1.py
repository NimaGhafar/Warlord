# agent1.py
import torch
from dqn_model import DQN
from agent_rl import RLAgent # We hergebruiken de klasse voor de structuur

class Agent1:
    def __init__(self):
        # Gebruik de juiste vorm van de observatie
        obs_shape = (210, 160, 3) 
        input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        num_actions = 3 # Hardcoded voor Warlords
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Maak een instantie van de agent, maar laad het getrainde model
        self.rl_agent = RLAgent(input_shape, num_actions, device)
        # Zorg ervoor dat het pad correct is!
        self.rl_agent.load_model('saved_models/warlords_dqn.pth')
        print("Agent1 (RL) geladen met getraind model.")

    def act(self, observation):
        # Gebruik de act-methode in evaluatiemodus (geen exploratie)
        return self.rl_agent.act(observation, evaluation_mode=True)