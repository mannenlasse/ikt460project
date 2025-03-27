import numpy as np


class Game: 

    def __init__(self, height, width):

        self.board_height = height
        self.board_width = width

        self.board = np.zeros((self.board_height, self.board_width), dtype=int)
        self.winning_length = 4
        self.Player1 = True  #Player 1 starts
        self.Winner = None  # Winner status (starting with none)





    def SimulateGame(self):
        print("ammar er gay. Yeet!")
        print(self.board)



game = Game(6,7)

game.SimulateGame()
