import random
from game import Game
from Agents.random_agent import RandomAgent
from Agents.agenta import Random_Agent_2
# Updated import path
from Agents.double_dqn.double_dqn_agent import DoubleDQNAgent
import os


board_height = 6
board_width = 7
number_of_players = 2
winnding_length = 3

# height --- width --- numvber of players --- winning length #
game = Game(board_height, board_width, number_of_players, winnding_length)

# Initialiser DQN-agenten og last inn trent modell
dqn_agent = DoubleDQNAgent(player_id=1, game=game,action_size=game.board_width)


# Update path in load_model call
dqn_agent.load_model("Agents/double_dqn/models/dqn_agent_final.pt")
dqn_agent.epsilon = 0.0  # Ingen utforskning, kun utnyttelse



agents = [dqn_agent, Random_Agent_2(1)]

print("main.py: Game started!\n")

done = False

while not done:

     
    
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
        done = True
        

    # Switch to next player
    game.current_player = (game.current_player % game.number_of_players) + 1
