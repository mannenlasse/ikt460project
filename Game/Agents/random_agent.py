import random
from .base_agent import Agent

class RandomAgent(Agent):
    def select_action(self, game):
        available_columns = [col for col in range(game.board_width) if game.board[0, col] == 0]

        if not available_columns:
            print("Random_Agent: random_available_column: no cells available")
            return None
        

        #chooses a randomn column that has an empty position 
        random_available_column = random.choice(available_columns)

        print(f"RandomAgent: chose: {random_available_column} as a random column")

        return random_available_column