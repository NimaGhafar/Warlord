import torch
from dqn_model import DQN
from agent_rl import RLAgent

class Agent1:
    def __init__(self):
        obs_shape = (210, 160, 3) 
        input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        num_actions = 3 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rl_agent = RLAgent(input_shape, num_actions, device)
        self.rl_agent.load_model('saved_models/warlords_dqn.pth')
        print("Agent1 (RL) geladen met getraind model.")

    def act(self, observation):
        return self.rl_agent.act(observation, evaluation_mode=True)