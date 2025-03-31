import random
from game import Game
 
height = 8
width = 12
num_players = 4
win_length = 4
game = Game(height, width, num_players, win_length)


def random_available_column(self):
    #a variable that goes through all the empty positions on the board(all positions in the array that are 0) 
    available_columns = [col for col in range(self.board_width) if self.board[0, col] == 0]
    
    if not available_columns:
        print("random_available_column: no cells available")
        return None
    
    #chooses a randomn column that has an empty position 
    random_available_column = random.choice(available_columns)

    print(f"random_available_column: chose: {random_available_column} as a random column")

    return random_available_column


print("Game started!\n")
done = False

while not done:
    move = random_available_column(game)
    if move is None:
        print("Board is full. It's a draw.")
        break

    result = game.make_move(move)
    if not result:
        continue  # Try again if move failed

    row, col = result
    print(f"Player {game.current_player} played in column {move}")
    game.print_board()
    print("")

    if game.winning_moves(row, col):
        print(f"Player {game.current_player} has won the game!\n")
        break

    # Switch to next player
    game.current_player = (game.current_player % game.number_of_players) + 1