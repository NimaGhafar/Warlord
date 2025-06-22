# train_ppo.py (Definitieve, robuuste versie)

import gymnasium as gym
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import supersuit as ss
import time

from pettingzoo.atari import warlords_v3
from ppo_agent import PPOAgent

def train_warlords_ppo(
    total_timesteps=500_000,
    update_interval=2048,
    lr=2.5e-4,
    gamma=0.99,
    eps_clip=0.2,
    k_epochs=4
):

   # 1. Setup van de omgeving
    env = warlords_v3.parallel_env(render_mode=None)
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, stack_size=4)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=255)

    # 2. Agenten initialiseren
    agents = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    for agent_id in env.possible_agents:
        obs_space = env.observation_space(agent_id)
        action_space = env.action_space(agent_id)
        agents[agent_id] = PPOAgent(
            obs_space_shape=obs_space.shape,
            action_space_n=action_space.n,
            lr=lr,
            gamma=gamma,
            eps_clip=eps_clip,
            k_epochs=k_epochs,
            device=device
        )

    # 3. Logging
    all_episode_rewards = []
    episode_count = 0
    obs, info = env.reset()

    # 4. Trainingslus
    print("Starting training...")
    for timestep in range(total_timesteps):
        active_agents = list(obs.keys())
        actions = {}

        for agent_id in active_agents:
            agent_obs = obs[agent_id]
            action, log_prob = agents[agent_id].choose_action(agent_obs)

            agents[agent_id].memory.states.append(agent_obs)
            agents[agent_id].memory.actions.append(action)
            agents[agent_id].memory.logprobs.append(log_prob)
            actions[agent_id] = action

        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        for agent_id in active_agents:
            done = terminations[agent_id] or truncations[agent_id]
            agents[agent_id].memory.rewards.append(rewards[agent_id])
            agents[agent_id].memory.dones.append(done)

        if any(terminations.values()) or any(truncations.values()):
            obs, info = env.reset()
            episode_count += 1
            print(f"Timestep: {timestep+1}/{total_timesteps}, Episode {episode_count} finished.")
        else:
            obs = next_obs

        if (timestep + 1) % update_interval == 0:
            for agent in agents.values():
                if len(agent.memory.states) > 0:
                    agent.update()

    env.close()
    print("Training voltooid!")
    return agents

if __name__ == "__main__":
    trained_agents = train_warlords_ppo()