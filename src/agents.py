# agents.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from utils import ReplayBuffer

class RandomAgent:
    """Een agent die willekeurige acties kiest uit de beschikbare actieruimte."""
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, state):
        return self.action_space.sample()

    # Lege methoden om compatibel te zijn met de trainingsloop
    def update(self, *args):
        pass

    def learn(self, *args):
        pass


class QNetwork(nn.Module):
    """
    Convolutional Neural Network voor het benaderen van Q-waardes uit schermpixels.
    """
    def __init__(self, observation_space_shape, action_space_n):
        super(QNetwork, self).__init__()
        # Input shape: (in_channels, H, W) -> (4, 84, 84) na preprocessing
        # We gaan uit van een gestapelde input van 4 frames (gebruikelijk in Atari)
        self.network = nn.Sequential(
            nn.Conv2d(observation_space_shape[2], 32, kernel_size=8, stride=4), # H, W, C -> C, H, W
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), # De input size hangt af van de output van de conv layers
            nn.ReLU(),
            nn.Linear(512, action_space_n)
        )

    def forward(self, x):
        # Permute de dimensies van (N, H, W, C) naar (N, C, H, W) voor PyTorch
        return self.network(x.permute(0, 3, 1, 2))


class DQNAgent:
    """Een Deep Q-Learning agent."""
    def __init__(self, action_space, observation_space,
                 learning_rate=1e-4,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.9995,
                 min_epsilon=0.1,
                 buffer_size=10000,
                 batch_size=64,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.device = device

        self.q_network = QNetwork(observation_space.shape, action_space.n).to(self.device)
        self.target_network = QNetwork(observation_space.shape, action_space.n).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size)

    def choose_action(self, state):
        """Kiest een actie met een epsilon-greedy strategie."""
        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        """Slaat een transitie op in de replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """
        Voert een leerstap uit:
        1. Sample een batch uit de replay buffer.
        2. Bereken de target Q-waardes.
        3. Voer een gradient descent stap uit op de Q-network.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        # Huidige Q-waardes voor de gekozen acties
        current_q_values = self.q_network(states).gather(1, actions)

        # Target Q-waardes
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            # Als de episode is afgelopen (done), is de target enkel de reward
            target_q_values = rewards + self.gamma * next_q_values * (~dones)

        # Bereken loss en update netwerk
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        """Verlaagt epsilon om minder te exploreren naarmate de training vordert."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Kopieert de gewichten van het q_network naar het target_network."""
        self.target_network.load_state_dict(self.q_network.state_dict())