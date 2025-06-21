import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import ReplayBuffer
import torch.nn.functional as F


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
        state_np = np.array(state, copy=False)               # nu écht een ndarray
        state_v  = torch.from_numpy(state_np)                # tensor dtype float32
        state_v  = state_v.unsqueeze(0).to(self.device)      # batchdim

        # Epsilon‑greedy
        self.steps_done += 1
        eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)

        if random.random() < eps_threshold:
            return random.randrange(self.q_net.net[-1].out_features)
        with torch.no_grad():
            q_vals = self.q_net(state_v / 255.0)
            return int(q_vals.max(1)[1].item())

    def learn(self, batch_size):
        # 1. Check of er genoeg samples in de buffer staan
        if len(self.replay_buffer) < batch_size:
            return

        # 2. Sample een batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # 3. Zet alles om naar PyTorch tensors
        states_v      = torch.tensor(states,      device=self.device, dtype=torch.float32)
        actions_v     = torch.tensor(actions,     device=self.device, dtype=torch.int64)
        rewards_v     = torch.tensor(rewards,     device=self.device, dtype=torch.float32)
        next_states_v = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones_v       = torch.tensor(dones,       device=self.device, dtype=torch.bool)

        # 4. Bereken current Q(s,a) values
        #    self.q_net(states_v) geeft (batch_size, n_actions)
        #    gather haalt de Q-waarde op voor de gekozen acties
        state_action_values = self.q_net(states_v).gather(
            1, actions_v.unsqueeze(-1)
        ).squeeze(-1)

        # 5. Bereken next Q-values via het target network (max_a' Q_target(s',a'))
        with torch.no_grad():
            next_state_values = self.target_net(next_states_v).max(1)[0]
            # Zet toekomstige waarden op 0 als de state terminal was
            next_state_values[dones_v] = 0.0

        # 6. Bellman-backup: y = r + γ * max Q_target(s', a')
        expected_state_action_values = rewards_v + self.gamma * next_state_values

        # 7. Loss berekenen (MSE tussen huidige en verwachte Q-waarden)
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # 8. Backpropagation & optimalisatie
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# --- Agent2: random baseline ---
class Agent2:
    def act(self, observation):
        # Return een willekeurige actie (6 mogelijk in ALE Warlords)
        return np.random.randint(6)