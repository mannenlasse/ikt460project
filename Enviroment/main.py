import numpy as np
import random

class Game: 

    def __init__(self, height, width):
        self.board_height = height
        self.board_width = width
        self.board = np.zeros((self.board_height, self.board_width), dtype=int)
        self.winning_length = 4
        self.current_player = 1  



    def make_move(self, index):
        for row in (reversed(range(self.board_height))):
            if self.board[row, index] == 0:
                self.board[row, index] = self.current_player
                return True  
        
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




    def winning_moves(self, index):
        print("winning_moves")
        



    def print_board(self):
        print(self.board)
        #print(np.flip(self.board))


    def simulate_game(self):
        print("SimulateGame: started\n")

        while True:
            col = self.random_available_column()
            if col is None:
                print("Board is full\n")
                break

            self.make_move(col)
            print(f"Player {self.current_player} played in column {col}")
          
            """
            # Get value at row 3, column 4
            value1 = self.board[0, 5]
            value2 = self.board[1, 5]
            value3 = self.board[2, 5]
            value4 = self.board[3, 5]
            value5 = self.board[4, 5]
            value6 = self.board[5, 5]
            print(f"The value1 in col: 6 row 1 is {value1} \n")
            print(f"The value2 in col: 6 row 2 {value2} \n")
            print(f"The value3 in col: 6 row 3 {value3} \n")
            print(f"The value4 in col: 6 row 4 {value4} \n")
            print(f"The value5 in col: 6 row 5 {value5} \n")
            print(f"The value6 in col: 6 row 6 {value6} \n")
            """

            self.print_board()
            print("")

            # Switch player (for now, between 1 and 2)
            self.current_player = 2 if self.current_player == 1 else 1


game = Game(6,8)

game.simulate_game()
