import random
from .base_agent import Agent
from print import log
import numpy as np

class RandomAgent(Agent):

    def __init__(self, Current_Player):
        self.current_player = Current_Player
        


    def select_action(self, game, player_id=None):

        
        available_columns = game.get_valid_columns()
        #available_columns = np.flatnonzero(game.board[0] == 0).tolist()
        if not available_columns:
            log("random_agent.py: no cells available")
            #print("random_agent.py: no cells available")

            return None
        
        random_available_column = random.choice(available_columns)
        #print(f"RandomAgent: chose column {random_available_column} for player {player_id}")
        log(f"RandomAgent: chose column {random_available_column} for player {player_id}")

        return random_available_column

