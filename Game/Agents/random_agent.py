import random
from .base_agent import Agent


class RandomAgent(Agent):

    def __init__(self, current_player):
        self.current_player = current_player
        



    def select_action(self, game):


        available_columns = [col for col in range(game.board_width) if game.board[0, col] == 0]

        if not available_columns:
            print("random_agent.py: Random_Agent: random_available_column: no cells available")
            return None
        


        #chooses a randomn column that has an empty position 
        random_available_column = random.choice(available_columns)

        print(f"random_agent.py: RandomAgent: chose: {random_available_column} as a random column for player {self.current_player}")

        return random_available_column
    

    def observe(self, reward, next_state, done):
    # Random agent doesn't learn, so we just pass
      print("no learning")
