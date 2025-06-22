import gymnasium as gym
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import supersuit as ss

from pettingzoo.atari import warlords_v3

from agents import DQNAgent, RandomAgent

def train_warlords(n_episodes=1000, use_dqn=True, learning_rate=1e-4,
                   target_update_freq=500, batch_size=32):
    """
    Train de MARL-agents op de Warlords-omgeving.
    """
    # 1. Omgeving opzetten
    env = warlords_v3.env(render_mode=None)

    env = ss.max_observation_v0(env, 2)

    env = ss.frame_skip_v0(env, 4)

    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, stack_size=4)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=255)

    # 2. Agents initialiseren
    env.reset()
    agents = {}
    episode_rewards_history = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    for agent_id in env.possible_agents:
        obs_space = env.observation_space(agent_id)
        action_space = env.action_space(agent_id)
        
        q_network_obs_shape = obs_space.shape
        
        if use_dqn:
            agents[agent_id] = DQNAgent(
                action_space=action_space,
                observation_space=gym.spaces.Box(low=0, high=1, shape=q_network_obs_shape, dtype=np.float32),
                learning_rate=learning_rate,
                batch_size=batch_size,
                device=device
            )
        else:
            agents[agent_id] = RandomAgent(action_space)

    global_step = 0

    for episode in range(n_episodes):
        env.reset()
        episode_rewards = defaultdict(float)
        
        previous_states = {}
        previous_actions = {}
        
        for agent_id in env.agent_iter():
            global_step += 1
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if agent_id in previous_states:
                prev_obs = previous_states[agent_id]
                prev_act = previous_actions[agent_id]
                agents[agent_id].update(prev_obs, prev_act, reward, observation, done)
                agents[agent_id].learn()

            if done:
                env.step(None)
            else:
                action = agents[agent_id].choose_action(observation)
                env.step(action)
                previous_states[agent_id] = observation
                previous_actions[agent_id] = action

            episode_rewards[agent_id] += reward
            
            if use_dqn and global_step % target_update_freq == 0:
                for agent in agents.values():
                    if isinstance(agent, DQNAgent):
                        agent.update_target_network()

        if use_dqn:
            for agent in agents.values():
                 if isinstance(agent, DQNAgent):
                    agent.decay_epsilon()
        
        avg_episode_reward = sum(episode_rewards.values()) / len(agents)
        episode_rewards_history.append(avg_episode_reward)

        current_epsilon = agents[env.possible_agents[0]].epsilon if use_dqn and isinstance(agents[env.possible_agents[0]], DQNAgent) else 'N/A'
        print(f"Episode {episode + 1}/{n_episodes} | Avg Reward: {avg_episode_reward:.2f} | Epsilon: {current_epsilon if isinstance(current_epsilon, str) else f'{current_epsilon:.4f}'}")

    env.close()

    plt.plot(episode_rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Average Total Reward per Agent")
    plt.title("MARL Training on Warlords")
    plt.savefig("training_rewards.png")
    plt.show()

    return agents

if __name__ == "__main__":
    trained_agents = train_warlords(n_episodes=5000, use_dqn=True)
    print("Training voltooid!")