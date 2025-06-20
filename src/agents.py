import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import ReplayBuffer


# --- Agent1: DQN-agent ---
class DQNNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0]*input_shape[1]*input_shape[2], 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        return self.net(x / 255.0)

class Agent1:
    def __init__(self, state_shape, n_actions, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=100000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DQNNet(state_shape, n_actions).to(self.device)
        self.target_net = DQNNet(state_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(100000)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action(self, state):
        self.steps_done += 1
        eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)
        if random.random() < eps_threshold:
            return random.randrange(self.q_net.net[-1].out_features)
        else:
            state_v = torch.tensor([state], device=self.device, dtype=torch.float32)
            q_vals = self.q_net(state_v)
            _, act_v = torch.max(q_vals, dim=1)
            return int(act_v.item())

    def learn(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        # TODO: zet hier tensor-conversies en Bellman-update
        pass

# --- Agent2: random baseline ---
class Agent2:
    def act(self, observation):
        # Return een willekeurige actie (6 mogelijk in ALE Warlords)
        return np.random.randint(6)