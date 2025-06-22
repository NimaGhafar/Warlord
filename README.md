# Multi-Agent PPO Training op Warlords

Dit project bevat een implementatie van Proximal Policy Optimization (PPO) voor multi-agent reinforcement learning in de Warlords-omgeving van PettingZoo. Het doel is om meerdere agents tegelijk te trainen in een visuele Atari-omgeving.


## Installatie

Om dit project te draaien, heb je de volgende Python-pakketten nodig:

```bash
pip install -r requirements.txt
````

## Gebruik

1. **Trainen van de PPO Agents**
   Het trainen van de PPO agents gebeurt door het uitvoeren van:

   ```python
   trained_agents = train_ppo.train_warlords_ppo(total_timesteps=60_000, update_interval=2048)
   ```

## Bestanden

* `ppo_agent.py`: Bevat de implementatie van de PPO-agent.
* `train_ppo.py`: Bevat de logica voor het trainen van de agents in de parallelle Warlord-omgeving.
* 'random_agent.py' : Bevat de implementatie van de Random-agent
* `env.py`: De PettingZoo Warlords-omgeving.
## Auteur

Dit project is gemaakt door Nima, Tommi, Vince en Isa
