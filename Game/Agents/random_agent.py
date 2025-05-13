import random
from .base_agent import Agent
from print import log

class RandomAgent(Agent):

    def __init__(self, Current_Player):
        self.current_player = Current_Player
        


    def select_action(self, game, player_id=None):
        available_columns = [col for col in range(game.board_width) if game.board[0, col] == 0]

        if not available_columns:
            log("random_agent.py: no cells available")
            #print("random_agent.py: no cells available")

            return None
        
        random_available_column = random.choice(available_columns)
        #print(f"RandomAgent: chose column {random_available_column} for player {player_id}")
        log(f"RandomAgent: chose column {random_available_column} for player {player_id}")

        return random_available_column

