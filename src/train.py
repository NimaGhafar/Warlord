"""
Trainings-entry-point voor:
  • baseline-run (RandomPolicy)
  • PPO-training
"""

import sys
import yaml

from src.env.warlords_env import make_env
from src.agents.agent import PPOAgent
from src.base.random_policy import RandomPolicy


def main(config_path: str):
    # 1. Config inladen
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    # 2. Omgeving initialiseren
    env = make_env(config.get("seed"))

    # Flatten-vector-dimensie (MLP-versie).
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 3. Baseline-run (RandomPolicy)
    baseline = RandomPolicy(env.action_space)

    obs = env.reset()
    for _ in range(1_000):
        action = baseline.select_action(obs)
        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()

    # 4. PPO-agent trainen
    agent = PPOAgent(obs_dim, action_dim, config["ppo"])

    obs = env.reset()
    for _ in range(int(config["ppo"]["max_timesteps"])):
        # a. actie kiezen
        action = agent.select_action(obs)

        # b. stap in omgeving
        obs, reward, done, info = env.step(action)

        # c. reward + done in buffer
        agent.store_reward(reward, done)

        # d. episode-einde → policy-update
        if done:
            agent.update()
            obs = env.reset()

    # 5. Model opslaan
    agent.save(config["ppo"]["save_path"])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Gebruik:  python train.py <config_file.yaml>")
    main(sys.argv[1])