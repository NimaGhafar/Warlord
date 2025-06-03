from src.agents.agent import PPOAgent
from pettingzoo.atari import warlords_v3
import numpy as np
import torch
from tqdm import trange

def train_ppo_agent(n_episodes=100, render=False):
    env = warlords_v3.env(render_mode="human" if render else None)
    env.reset()

    # Kies de agent die je traint
    train_agent_name = env.agents[0]  # bijv. 'first_0'
    
    # Verkrijg de shape en actieruimte van die agent
    obs_shape = env.observation_spaces[train_agent_name].shape
    act_dim = env.action_spaces[train_agent_name].n

    config = {
        'gamma': 0.99,
        'eps_clip': 0.2,
        'K_epochs': 4,
        'lr': 3e-4
    }

    agent = PPOAgent(np.prod(obs_shape), act_dim, config)
    rewards_per_episode = []

    for episode in trange(n_episodes, desc="Training"):
        env.reset()
        total_reward = 0
        done = False
        state = None

        for step in env.agent_iter():
            current_agent = env.agent_selection
            obs, reward, termination, truncation, info = env.last()

            if current_agent == train_agent_name:
                done = termination or truncation
                if done or obs is None:
                    env.step(None)
                else:
                    flat_obs = np.array(obs).flatten() / 255.0
                    action = agent.select_action(flat_obs)
                    agent.buffer['rewards'][-1] = reward
                    agent.buffer['dones'].append(done)
                    total_reward += reward
                    env.step(action)
                    state = flat_obs
            else:
                # Random actie voor andere agenten (alleen als ze nog leven)
                if termination or truncation:
                    env.step(None)
                else:
                    env.step(env.action_spaces[current_agent].sample())

                    agent.update()
        rewards_per_episode.append(total_reward)
        print(f"Episode {episode+1}/{n_episodes}, Reward: {total_reward}")

    env.close()
    return rewards_per_episode, agent
    
if __name__ == "__main__":
    episodes = 500
    rewards, agent = train_ppo_agent(n_episodes=episodes, use_random=False, render=False, save_path="ppo_warlords.pth")

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Reward per Episode")
    plt.grid(True)
    plt.show()