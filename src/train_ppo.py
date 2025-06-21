# train_ppo.py

import gymnasium as gym
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import supersuit as ss
import time

from pettingzoo.atari import warlords_v3
from ppo_agent import PPOAgent

def train_warlords_ppo(total_timesteps=1_000_000, update_interval=2048):
    """
    Train MARL agents op de Warlords-omgeving met PPO.
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    for agent_id in env.possible_agents:
        obs_space = env.observation_space(agent_id)
        action_space = env.action_space(agent_id)
        agents[agent_id] = PPOAgent(
            obs_space_shape=obs_space.shape,
            action_space_n=action_space.n,
            device=device
        )

    # Logging
    episode_rewards_history = []
    current_episode_rewards = defaultdict(float)
    start_time = time.time()

    # 3. Trainingsloop
    env.reset()
    for timestep in range(total_timesteps):
        # Kies een actie voor elke agent
        actions = {}
        for agent_id in env.agents:
            state = env.observe(agent_id)
            action, log_prob = agents[agent_id].choose_action(state)
            
            # Sla data op in het geheugen van de agent
            agents[agent_id].memory.states.append(state)
            agents[agent_id].memory.actions.append(action)
            agents[agent_id].memory.logprobs.append(log_prob)
            
            actions[agent_id] = action

        # Voer stap uit in omgeving (parallel API is makkelijker voor PPO)
        # We moeten de AEC env converteren naar parallel
        # Dit is een vereenvoudiging voor de PPO-loop
        
        # De PettingZoo AEC-API step-logica voor PPO
        for agent_id in env.agents:
            if env.terminations[agent_id] or env.truncations[agent_id]:
                env.step(None) # Agent is done, stuur 'None' actie
            else:
                env.step(actions[agent_id])

        # Haal rewards en dones op
        rewards = env.rewards
        dones = {a: env.terminations[a] or env.truncations[a] for a in env.agents}

        # Sla rewards en dones op in geheugen
        for agent_id in env.agents:
            agents[agent_id].memory.rewards.append(rewards[agent_id])
            agents[agent_id].memory.dones.append(dones[agent_id])
            current_episode_rewards[agent_id] += rewards[agent_id]

        # Als een episode voorbij is voor alle agents, log de rewards
        if all(dones.values()):
            avg_reward = sum(current_episode_rewards.values()) / len(agents)
            episode_rewards_history.append(avg_reward)
            print(f"Timestep: {timestep}/{total_timesteps}, Avg Reward: {avg_reward:.2f}")
            current_episode_rewards.clear()
            env.reset()

        # Update de policy als het tijd is
        if (timestep + 1) % update_interval == 0:
            print(f"\n--- Updating policy at timestep {timestep} ---")
            for agent in agents.values():
                agent.update()
            
            # Print performance
            time_elapsed = time.time() - start_time
            steps_per_second = (timestep + 1) / time_elapsed
            print(f"Steps/sec: {steps_per_second:.2f}\n")


    env.close()

    # Plot de resultaten
    plt.plot(episode_rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Average Total Reward per Agent")
    plt.title("MARL PPO Training on Warlords")
    plt.show()

    return agents

if __name__ == "__main__":
    trained_agents = train_warlords_ppo()
    print("Training voltooid!")