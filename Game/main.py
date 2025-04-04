import random
from game import Game
from Agents.random_agent import RandomAgent 

height = 8
width = 12
num_players = 2
win_length = 4
game = Game(height, width, num_players, win_length)


agents = [
    RandomAgent(1),       # Player 1 uses RandomAgent
    RandomAgent(2)     # Player 2 uses NotRandomAgent
]

print("main.py: Game started!\n")
done = False

while not done:

     
    #current_agent = agents[game.current_player - 1]
    #print(f"main.py: current agent: {current_agent} and agents[game.current_player - 1]: {agents[game.current_player - 1]} and agents: {agents}")
    #move = current_agent.select_move_random(game)
    
    print(f"Current player: {game.current_player}")


    current_agent = agents[game.current_player - 1]

    move = current_agent.select_action(game)
    


    if move is None:
        print("main.py: Board is full. It's a draw.")
        break

    result = game.make_move(move)
    if not result:
        continue  # Try again if move failed

    row, col = result
    print(f"main.py: Player {game.current_player} played in column {move} and col: {col} and row: {row} and result: {result}")
    game.print_board()
    print("")

    if game.winning_moves(row, col):
        print(f"main.py: Player {game.current_player} has won the game!\n")
        break

    # Switch to next player
    game.current_player = (game.current_player % game.number_of_players) + 1
