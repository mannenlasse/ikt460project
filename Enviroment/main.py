import numpy as np
import random

class Game: 

    def __init__(self, height, width):
        self.board_height = height
        self.board_width = width

        self.board = np.zeros((self.board_height, self.board_width), dtype=int)
        self.winning_length = 4
        self.current_player = 1  



    def make_move(self, col):
        for row in reversed(range(self.board_height)):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                return True  # <--- ADD THIS!
        
        print("make_move: no available moves")
        return False



    def random_available_column(self):
        available_columns = [col for col in range(self.board_width) if self.board[0, col] == 0]
        if not available_columns:
            print("random_available_column: no cells available")
            return None
        
        random_move = random.choice(available_columns)

        print(f"random_available_column: chose: {random_move} as a random column")

        
        return random_move

    def print_board(self):
        print(np.flip(self.board))


    def simulate_game(self):
        print("SimulateGame: started\n")

        while True:
            col = self.random_available_column()
            if col is None:
                print("Board is full\n")
                break

            self.make_move(col)
            print(f"Player {self.current_player} played in column {col}")
            self.print_board()
            print("")

            # Switch player (for now, between 1 and 2)
            self.current_player = 2 if self.current_player == 1 else 1


game = Game(6,9)

game.simulate_game()
