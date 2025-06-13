"""
Agents voor Atari Warlords:
  • RandomPolicy (baseline)
  • PPOAgent (actor–critic met rollout-buffer)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from src.utils import device 

# Helper: lege rollout-buffer
def roll_out_buffer():
    return {
        "states":  [],
        "actions": [],
        "logprobs": [],
        "rewards": [],
        "dones":   [],
    }

# Actor–Critic netwerk (MLP-variant)
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Plaats het netwerk op het geselecteerde device
        self.to(device)

    # ---------- selectie van een enkele actie ---------- #
    def act(self, state):
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)

        probs  = self.actor(state_tensor)
        dist   = Categorical(probs)
        action = dist.sample()
        value  = self.critic(state_tensor)

        return action.item(), dist.log_prob(action), value.squeeze(0)

    # ---------- evaluatie tijdens de PPO-update ---------- #
    def evaluate(self, states, actions):
        states  = states.to(device)
        actions = actions.to(device)

        probs    = self.actor(states)
        dist     = Categorical(probs)
        logp     = dist.log_prob(actions)
        entropy  = dist.entropy()
        values   = self.critic(states).squeeze(-1)

        return logp, values, entropy


# --------------------------------------------------------------------------- #
# Random baseline-agent
# --------------------------------------------------------------------------- #
class RandomAgent:
    def act(self, observation):
        return np.random.randint(6)  # 6 mogelijke acties in ALE Warlords


# --------------------------------------------------------------------------- #
# PPO-agent
# --------------------------------------------------------------------------- #
class PPOAgent:
    def __init__(self, state_dim, action_dim, config: dict):
        # Hyper-parameters
        self.gamma    = config["gamma"]
        self.eps_clip = config["eps_clip"]
        self.K_epochs = config["K_epochs"]

        # Twee policies: current vs. old
        self.policy     = ActorCritic(state_dim, action_dim)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Optimizer & losses
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config["lr"]
        )
        self.MseLoss = nn.MSELoss()

        # Rollout-buffer
        self.buffer = roll_out_buffer()

    # ---------- actie kiezen ---------- #
    def select_action(self, state):
        action, logprob, _ = self.policy_old.act(state)

        self.buffer["states"].append(state)
        self.buffer["actions"].append(action)
        self.buffer["logprobs"].append(logprob)

        return action

    # ---------- reward + done opslaan ---------- #
    def store_reward(self, reward, done):
        self.buffer["rewards"].append(reward)
        self.buffer["dones"].append(done)

    # ---------- PPO-update ---------- #
    def update(self):
        # 1. Discounted returns
        returns = []
        discounted = 0.0
        for reward, done in zip(
            reversed(self.buffer["rewards"]),
            reversed(self.buffer["dones"]),
        ):
            if done:
                discounted = 0.0
            discounted = reward + self.gamma * discounted
            returns.insert(0, discounted)

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # 2. Buffer → tensors
        states = torch.tensor(
            self.buffer["states"], dtype=torch.float32, device=device
        )
        actions = torch.tensor(self.buffer["actions"], device=device)
        old_logprobs = (
            torch.stack(self.buffer["logprobs"]).detach().to(device)
        )

        # 3. Optimaliseer
        for _ in range(self.K_epochs):
            logprobs, values, entropy = self.policy.evaluate(states, actions)
            ratios = torch.exp(logprobs - old_logprobs)

            advantages = returns - values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(
                ratios, 1 - self.eps_clip, 1 + self.eps_clip
            ) * advantages

            loss = (
                -torch.min(surr1, surr2)  # policy-loss
                + 0.5 * self.MseLoss(values, returns)  # value-loss
                - 0.01 * entropy  # entropy-bonus
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 4. Synchroniseer oude policy + reset buffer
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = roll_out_buffer()

    # ---------- opslaan / laden ---------- #
    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(state_dict)
        self.policy_old.load_state_dict(state_dict)

        # Zorg dat geladen gewichten op het juiste device staan
        self.policy.to(device)
        self.policy_old.to(device)