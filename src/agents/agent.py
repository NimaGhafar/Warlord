import torch
import torch.nn as nn
from torch.distributions import Categorical

# Buffer om ervaringen op te slaan
def roll_out_buffer():
    return {
        "states": [],
        "actions": [],
        "logprobs": [],
        "rewards": [],
        "dones": []
    }

# Actor-Critic netwerk
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Actor: bepaalt welke actie te nemen
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim), nn.Softmax(dim=-1)
        )
        # Critic: schat de waarde van een staat
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), self.critic(state)

    def evaluate(self, states, actions):
        probs = self.actor(states)
        dist = Categorical(probs)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states).squeeze(-1)
        return logprobs, values, entropy

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.gamma = config['gamma']
        self.eps_clip = config['eps_clip']
        self.K_epochs = config['K_epochs']
        self.policy = ActorCritic(state_dim, action_dim)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['lr'])
        self.buffer = roll_out_buffer()
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        action, logprob, value = self.policy_old.act(state)
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['logprobs'].append(logprob)
        self.buffer['rewards'].append(value)
        return action

    def update(self):
        # Bereken discounted rewards
        rewards = []
        discounted = 0
        for reward, done in zip(reversed(self.buffer['rewards']), reversed(self.buffer['dones'])):
            discounted = 0 if done else discounted * self.gamma
            discounted += reward
            rewards.insert(0, discounted)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Verzamel bufferdata
        states = torch.FloatTensor(self.buffer['states'])
        actions = torch.tensor(self.buffer['actions'])
        old_logprobs = torch.stack(self.buffer['logprobs'])

        # Train de policy
        for _ in range(self.K_epochs):
            logprobs, values, entropy = self.policy.evaluate(states, actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values, rewards) - 0.01 * entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Update oude policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = roll_out_buffer()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy_old.load_state_dict(torch.load(path))