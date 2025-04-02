import random
from game import Game
from Agents.random_agent import RandomAgent 
height = 8
width = 12
num_players = 4
win_length = 4
game = Game(height, width, num_players, win_length)


agents = [RandomAgent(i + 1) for i in range(num_players)]

print("main.py: Game started!\n")
done = False

while not done:

    current_agent = agents[game.current_player - 1]
    move = current_agent.select_action(game)


    if move is None:
        print("main.py: Board is full. It's a draw.")
        break

    result = game.make_move(move)
    if not result:
        continue  # Try again if move failed

    row, col = result
    print(f"main.py: Player {game.current_player} played in column {move}")
    game.print_board()
    print("")

    if game.winning_moves(row, col):
        print(f"main.py: Player {game.current_player} has won the game!\n")
        break

    # Switch to next player
    game.current_player = (game.current_player % game.number_of_players) + 1