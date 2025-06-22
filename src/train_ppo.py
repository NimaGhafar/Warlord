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

def train_warlords_ppo(total_timesteps=500_000, update_interval=2048):
    """
    Train MARL agents op de Warlords-omgeving met PPO via de Parallel API.
    Deze versie bevat een robuuste dataverzamelingslus die size-mismatch errors voorkomt.
    """
    # 1. Omgeving opzetten met de parallel_env functie
    env = warlords_v3.parallel_env(render_mode=None)
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, stack_size=4)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=255)

    # 2. Agents initialiseren
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
    all_episode_rewards = []
    episode_count = 0
    
    # 3. Definitieve, robuuste trainingsloop
    print("Starting training...")
    obs, info = env.reset()
    
    for timestep in range(total_timesteps):
        # We bepalen de set van actieve agenten aan het begin van de stap.
        # Dit is de 'ground truth' voor deze timestep.
        active_agents = list(obs.keys())
        
        # Verzamel acties en sla de 'pre-step' data (state, log_prob) op.
        actions = {}
        for agent_id in active_agents:
            agent_obs = obs[agent_id]
            action, log_prob = agents[agent_id].choose_action(agent_obs)
            
            agents[agent_id].memory.states.append(agent_obs)
            agents[agent_id].memory.actions.append(action)
            agents[agent_id].memory.logprobs.append(log_prob)
            actions[agent_id] = action

        # Voer een stap uit met de acties van de actieve agenten.
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Sla de 'post-step' data (reward, done) op voor DEZELFDE set actieve agenten.
        # Dit garandeert dat de lijsten in `memory` synchroon blijven.
        for agent_id in active_agents:
            done = terminations[agent_id] or truncations[agent_id]
            agents[agent_id].memory.rewards.append(rewards[agent_id])
            agents[agent_id].memory.dones.append(done)

        # Als enige agent 'done' is, is de episode voorbij en resetten we.
        # De `next_obs` van een 'done' agent is niet meer geldig.
        if any(terminations.values()) or any(truncations.values()):
            obs, info = env.reset()
            episode_count += 1
            print(f"Timestep: {timestep+1}/{total_timesteps}, Episode {episode_count} finished.")
        else:
            # Anders gaan we gewoon door met de nieuwe observaties.
            obs = next_obs

        # Update de policy als het tijd is
        if (timestep + 1) % update_interval == 0:
            for agent in agents.values():
                if len(agent.memory.states) > 0:
                    agent.update()
            


    env.close()
    print("Training voltooid!")
    return agents

if __name__ == "__main__":
    trained_agents = train_warlords_ppo()