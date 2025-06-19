# agent_rl.py
import torch
import numpy as np
import random
from dqn_model import DQN

class RLAgent:
    def __init__(self, input_shape, num_actions, device, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=10000):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
        
        self.policy_net = DQN(input_shape, num_actions).to(device)
        self.target_net = DQN(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net is alleen voor evaluatie

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.steps_done = 0

    def act(self, state, evaluation_mode=False):
        """Kies een actie met epsilon-greedy strategie."""
        # Epsilon decay
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        if random.random() > self.epsilon or evaluation_mode:
            with torch.no_grad():
                # Permute de state van (H, W, C) naar (C, H, W) voor PyTorch
                state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.num_actions)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())