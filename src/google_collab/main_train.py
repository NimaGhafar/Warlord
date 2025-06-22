import torch
import gymnasium as gym
from pettingzoo.atari import warlords_v3
from replay_buffer import ReplayBuffer
from agent_rl import RLAgent
import numpy as np
from collections import defaultdict
import os

# Hyperparameters 
EPISODES = 5000
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 30000
TARGET_UPDATE = 1000 
REPLAY_BUFFER_SIZE = 100000
LEARNING_RATE = 0.0001
SAVE_DIR = "saved_models"

os.makedirs(SAVE_DIR, exist_ok=True)

env = warlords_v3.parallel_env(render_mode=None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

obs_space_shape = env.observation_space('first_0').shape
input_shape = (obs_space_shape[2], obs_space_shape[0], obs_space_shape[1]) 
num_actions = env.action_space('first_0').n

agents = {agent_id: RLAgent(input_shape, num_actions, device, gamma=GAMMA, epsilon_start=EPS_START, epsilon_end=EPS_END, epsilon_decay=EPS_DECAY) for agent_id in env.possible_agents}
replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

total_steps = 0
episode_rewards = defaultdict(list)

for episode in range(EPISODES):
    observations, infos = env.reset()

    current_episode_rewards = defaultdict(float)
    
    while env.agents: 
        actions = {}
        for agent_id in env.agents:
            obs_for_agent = observations[agent_id]
            actions[agent_id] = agents[agent_id].act(obs_for_agent)

        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent_id in env.agents:
            state = observations[agent_id]
            action = actions[agent_id]
            reward = rewards[agent_id]
            next_state = next_observations[agent_id]
            done = terminations[agent_id] or truncations[agent_id]
            
            replay_buffer.push(state, action, reward, next_state, done)
            current_episode_rewards[agent_id] += reward

        observations = next_observations
        total_steps += 1

        if len(replay_buffer) > BATCH_SIZE:
            for agent_id in agents:
                states, actions, rews, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(device)
                next_states = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(device)
                actions = torch.LongTensor(actions).to(device)
                rews = torch.FloatTensor(rews).to(device)
                dones = torch.FloatTensor(dones).to(device)

                current_q_values = agents[agent_id].policy_net(states).gather(1, actions.unsqueeze(1))
                next_q_values = agents[agent_id].target_net(next_states).max(1)[0].detach()

                expected_q_values = rews + (agents[agent_id].gamma * next_q_values * (1 - dones))

                loss = torch.nn.functional.mse_loss(current_q_values.squeeze(), expected_q_values)

                agents[agent_id].optimizer.zero_grad()
                loss.backward()
                agents[agent_id].optimizer.step()

        if total_steps % TARGET_UPDATE == 0:
            for agent_id in agents:
                agents[agent_id].target_net.load_state_dict(agents[agent_id].policy_net.state_dict())

    avg_reward = np.mean([val for val in current_episode_rewards.values()])
    print(f"Episode {episode + 1}/{EPISODES}, Total Steps: {total_steps}, Avg Reward: {avg_reward:.2f}, Epsilon: {agents['first_0'].epsilon:.2f}")

torch.save(agents['first_0'].policy_net.state_dict(), os.path.join(SAVE_DIR, "warlords_dqn.pth"))
print("Training voltooid en model opgeslagen.")
env.close()