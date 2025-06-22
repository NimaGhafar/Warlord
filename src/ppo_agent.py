import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, observation_space_shape, action_space_n):
        super(ActorCritic, self).__init__()

        self.cnn_base = nn.Sequential(
            nn.Conv2d(observation_space_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        dummy_input = torch.zeros(1, *observation_space_shape).permute(0, 3, 1, 2)
        cnn_out_size = self.cnn_base(dummy_input).shape[1]

        self.actor = nn.Sequential(
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_space_n),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, state):
        state = state.permute(0, 3, 1, 2)
        features = self.cnn_base(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

class Memory:
    """Geheugenbuffer om transities op te slaan voor een PPO-update."""
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]

class PPOAgent:
    def __init__(self, obs_space_shape, action_space_n, lr=2.5e-4, gamma=0.99,
                 k_epochs=4, eps_clip=0.2, device='cuda'):
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = device

        self.policy = ActorCritic(obs_space_shape, action_space_n).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(obs_space_shape, action_space_n).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss_fn = nn.MSELoss()
        self.memory = Memory()

    def choose_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, _ = self.policy_old(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()

    def update(self):
        rewards_to_go = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_to_go.insert(0, discounted_reward)
            
        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32).to(self.device)
        rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-5)

        old_states = torch.FloatTensor(np.array(self.memory.states)).to(self.device).detach()
        old_actions = torch.LongTensor(self.memory.actions).to(self.device).detach()
        old_logprobs = torch.FloatTensor(self.memory.logprobs).to(self.device).detach()

        for _ in range(self.k_epochs):
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)
            new_logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()

            ratios = torch.exp(new_logprobs - old_logprobs.detach())

            advantages = rewards_to_go - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.loss_fn(state_values, rewards_to_go.unsqueeze(1))
            entropy_loss = -0.01 * entropy.mean()
            
            loss = policy_loss + 0.5 * value_loss + entropy_loss

            # Gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()