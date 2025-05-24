import yaml
from src.env.warlords_env import make_env
from src.agents.agent import PPOAgent
from src/base/random_policy import RandomPolicy

def main(config_path):
    config = yaml.safe_load(open(config_path))
    env = make_env(config.get('seed'))
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Baseline run
    baseline = RandomPolicy(env.action_space)
    obs = env.reset()
    for _ in range(1000):
        action = baseline.select_action(obs)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    # PPO-agent initialiseren
    agent = PPOAgent(obs_dim, action_dim, config['ppo'])
    obs = env.reset()
    for t in range(int(config['ppo']['max_timesteps'])):
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        agent.store_reward(reward, done)
        if done:
            agent.update()
            obs = env.reset()

    # Model opslaan
    agent.save(config['ppo']['save_path'])

if __name__ == '__main__':
    import sys
    main(sys.argv[1])