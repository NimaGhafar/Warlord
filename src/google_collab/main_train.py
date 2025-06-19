# main_train.py
import torch
import gymnasium as gym
from pettingzoo.atari import warlords_v3
from replay_buffer import ReplayBuffer
from agent_rl import RLAgent
import numpy as np
from collections import defaultdict
import os

# --- Hyperparameters ---
EPISODES = 5000
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 30000
TARGET_UPDATE = 1000 # Hoe vaak het target netwerk updaten
REPLAY_BUFFER_SIZE = 100000
LEARNING_RATE = 0.0001
SAVE_DIR = "saved_models"

os.makedirs(SAVE_DIR, exist_ok=True)

# --- Initialisatie ---
env = warlords_v3.parallel_env(render_mode=None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Observatie- en actieruimte
obs_space_shape = env.observation_space('first_0').shape
# Permute naar (C, H, W)
input_shape = (obs_space_shape[2], obs_space_shape[0], obs_space_shape[1]) 
num_actions = env.action_space('first_0').n

# Maak 4 onafhankelijke agenten (IQL)
agents = {agent_id: RLAgent(input_shape, num_actions, device, gamma=GAMMA, epsilon_start=EPS_START, epsilon_end=EPS_END, epsilon_decay=EPS_DECAY) for agent_id in env.possible_agents}
replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

total_steps = 0
episode_rewards = defaultdict(list)

# --- Trainingsloop ---
for episode in range(EPISODES):
    observations, infos = env.reset()
    
    # Track de totale reward voor deze episode per agent
    current_episode_rewards = defaultdict(float)
    
    while env.agents: # Loop zolang er agenten in de game zijn
        actions = {}
        for agent_id in env.agents:
            # We moeten de observatie in het juiste formaat voor de agent's act-methode zetten
            obs_for_agent = observations[agent_id]
            actions[agent_id] = agents[agent_id].act(obs_for_agent)

        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Sla de experience op in de replay buffer
        for agent_id in env.agents:
            state = observations[agent_id]
            action = actions[agent_id]
            reward = rewards[agent_id]
            next_state = next_observations[agent_id]
            # 'done' is waar als de agent uit het spel is (terminated) of de game eindigt (truncated)
            done = terminations[agent_id] or truncations[agent_id]
            
            replay_buffer.push(state, action, reward, next_state, done)
            current_episode_rewards[agent_id] += reward

        observations = next_observations
        total_steps += 1

        # --- Leerfase ---
        if len(replay_buffer) > BATCH_SIZE:
            # Voor elke agent, doe een leerstap
            for agent_id in agents:
                # We kunnen de buffer delen of aparte buffers gebruiken. Hier gebruiken we een gedeelde buffer.
                states, actions, rews, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                
                # Converteer naar tensors
                states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(device)
                next_states = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(device)
                actions = torch.LongTensor(actions).to(device)
                rews = torch.FloatTensor(rews).to(device)
                dones = torch.FloatTensor(dones).to(device)

                # Bereken Q-waardes
                current_q_values = agents[agent_id].policy_net(states).gather(1, actions.unsqueeze(1))
                next_q_values = agents[agent_id].target_net(next_states).max(1)[0].detach()
                
                # Bereken de expected Q-waardes (Bellman equation)
                expected_q_values = rews + (agents[agent_id].gamma * next_q_values * (1 - dones))

                # Bereken loss
                loss = torch.nn.functional.mse_loss(current_q_values.squeeze(), expected_q_values)

                # Optimaliseer het model
                agents[agent_id].optimizer.zero_grad()
                loss.backward()
                agents[agent_id].optimizer.step()

        # Update het target netwerk periodiek
        if total_steps % TARGET_UPDATE == 0:
            for agent_id in agents:
                agents[agent_id].target_net.load_state_dict(agents[agent_id].policy_net.state_dict())

    # Logging na elke episode
    avg_reward = np.mean([val for val in current_episode_rewards.values()])
    print(f"Episode {episode + 1}/{EPISODES}, Total Steps: {total_steps}, Avg Reward: {avg_reward:.2f}, Epsilon: {agents['first_0'].epsilon:.2f}")

# Sla het getrainde model van één agent op (ze zijn allemaal identiek getraind)
torch.save(agents['first_0'].policy_net.state_dict(), os.path.join(SAVE_DIR, "warlords_dqn.pth"))
print("Training voltooid en model opgeslagen.")
env.close()