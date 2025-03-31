import numpy as np
import random

class Game: 

    def __init__(self, height, width, num_players, win_length):
        self.board_height = height
        self.board_width = width
        self.number_of_players = num_players
        self.board = np.zeros((self.board_height, self.board_width), dtype=int)
        self.winning_length = win_length
        self.current_player = 1  
        self.winner = None



    def make_move(self, index):
        for row in (reversed(range(self.board_height))):
            if self.board[row, index] == 0:
                self.board[row, index] = self.current_player
                return row, index  
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


    def winning_moves(self, row, col):
        player = self.current_player  # more direct and clearer
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]  # horiz, vert, diag↘, diag↗

        for dr, dc in directions:
            count = 1

            # Check one direction
            r, c = row + dr, col + dc
            while 0 <= r < self.board_height and 0 <= c < self.board_width and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc

            # Check the opposite direction
            r, c = row - dr, col - dc
            while 0 <= r < self.board_height and 0 <= c < self.board_width and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= self.winning_length:
                self.winner = player
                print(f"Player {player} wins with {count} in a row!")
                return True

        return False


    def print_board(self):
        print(self.board)
        #print(np.flip(self.board))


    def simulate_game(self):
        print("SimulateGame: started\n")

        condition = True 
        
        while condition:
            
            move = self.random_available_column()

            if move is None:
                print("Board is full\n")
                break

            row_col = self.make_move(move)

            row, col = row_col
            print(f"Player {self.current_player} played in column {move}")

            if self.winning_moves(row, col):
                print(f"Player {self.current_player} has won!!!")
                condition = False
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
            self.current_player = (self.current_player % self.number_of_players) + 1



game = Game(6,7,2,4)

game.simulate_game()
