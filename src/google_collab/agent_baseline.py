# agent_baseline.py
import numpy as np

class BaselineAgent:
    """Een simpele rule-based agent voor Warlords."""
    def __init__(self, agent_id=0):
        # agent_id is 0, 1, 2, of 3, wat de hoek bepaalt
        self.agent_id = agent_id
        # We weten uit de documentatie dat de actieruimte 3 is (stil, links, rechts)
        self.action_space_size = 3 

    def find_ball(self, observation):
        """Zoekt naar het witte object (de bal) in de observatie."""
        # De bal is wit (kleurwaarde ~236)
        # We zoeken naar de coördinaten van de bal
        ball_color = [236, 236, 236]
        indices = np.where(np.all(observation == ball_color, axis=-1))
        if len(indices[0]) > 0:
            # Neem het gemiddelde van de gevonden coördinaten
            return np.mean(indices[1]), np.mean(indices[0]) # x, y
        return None, None

    def act(self, observation):
        """Kies een actie op basis van de balpositie."""
        ball_x, _ = self.find_ball(observation)

        if ball_x is None:
            return 0 # Doe niets als er geen bal is

        # De paddle-positie is moeilijk te bepalen, dus we gebruiken een benadering.
        # De breedte van het scherm is 210.
        screen_center_x = 210 / 2
        
        # Vereenvoudigde logica: als de bal links van het midden is, beweeg links, anders rechts.
        # Dit is een zeer simpele heuristiek!
        if ball_x < screen_center_x - 10:
            return 1 # Beweeg links
        elif ball_x > screen_center_x + 10:
            return 2 # Beweeg rechts
        else:
            return 0 # Blijf stil