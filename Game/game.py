import numpy as np
import random

class Game: 

    def __init__(self, height, width, num_players, win_length):
        #self. means current state    

        self.board_height = height
        self.board_width = width
        self.number_of_players = num_players

        #the board is a numpy array consisting of int and its size is defined by variables height and width
        #empty board consists of only 0 
        self.board = np.zeros((self.board_height, self.board_width), dtype=int)

        #variable defining how many in a row you need to win 
        self.winning_length = win_length

        #always start with players number 1 
        self.current_player = 1  

        self.winner = None



    def make_move(self, index):
        #since four in a row is with gravity. We use reverse to go through the board upside down so that it searches downwards and acts like gravity exsists

        for row in (reversed(range(self.board_height))):
            if self.board[row, index] == 0:
                self.board[row, index] = self.current_player
                return row, index  
        print("game.py: make_move: no available moves")
        return False


    def winning_moves(self, row, col):

        #All possitble directions: horixontal , vertical , diagonal up , diagonal down
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]  # horiz, vert, diag↘, diag↗

        for dr, dc in directions:
            count = 1

            # Check one direction
            r, c = row + dr, col + dc
            while 0 <= r < self.board_height and 0 <= c < self.board_width and self.board[r, c] == self.current_player:
                count += 1
                r += dr
                c += dc

            # Check the opposite direction
            r, c = row - dr, col - dc
            while 0 <= r < self.board_height and 0 <= c < self.board_width and self.board[r, c] == self.current_player:
                count += 1
                r -= dr
                c -= dc

            #if the current count is as same or more than whats needed to win the game, the current players becomes the winner. 
            if count >= self.winning_length:
                self.winner = self.current_player
                print(f"game.py: winning_moves: Player {self.current_player} wins with {count} in a row!")
                return True

        return False


    def print_board(self):
        print(self.board)
        #print(np.flip(self.board))

    def get_valid_columns(self):
        return [col for col in range(self.board_width) if self.board[0, col] == 0]
