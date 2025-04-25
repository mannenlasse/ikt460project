import sys
import os

# Add project root for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Game.game import Game
# Updated import path (relative)
from .double_dqn_agent import DoubleDQNAgent
from Game.Agents.random_agent import RandomAgent
import numpy as np
import torch # Added

# Game parameters
BOARD_HEIGHT = 6
BOARD_WIDTH = 7
NUM_PLAYERS = 2
WIN_LENGTH = 4
NUM_TEST_GAMES = 100

# Initialize agents
dqn_agent = DoubleDQNAgent(
    player_id=1,
    board_height=BOARD_HEIGHT,
    board_width=BOARD_WIDTH,
    action_size=BOARD_WIDTH,
    # Specify reward_type if needed for initialization, though it might not affect evaluation
    reward_type='sparse' # Or 'shaped', or load from model checkpoint if saved
)

# Load the trained model (Update path)
# Choose which model to test (sparse or shaped)
MODEL_FILENAME = "dqn_agent_sparse_final.pt" # Or "dqn_agent_shaped_final.pt"
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', MODEL_FILENAME)
if os.path.exists(MODEL_PATH):
     dqn_agent.load_model(MODEL_PATH)
else:
     print(f"Error: Model file not found at {MODEL_PATH}")
     sys.exit(1)

dqn_agent.epsilon = 0.0  # No exploration during testing

# Initialize opponent
opponent = RandomAgent(2) # Assign player ID 2

# Statistics
wins = 0
losses = 0
draws = 0

# Test loop
for game_num in range(NUM_TEST_GAMES):
    # Create a new game
    game = Game(BOARD_HEIGHT, BOARD_WIDTH, NUM_PLAYERS, WIN_LENGTH)
    
    # Game loop
    done = False
    while not done:
        current_player_id = game.current_player
        
        # Select action based on current player
        if current_player_id == 1:  # DQN agent's turn
            action = dqn_agent.select_action(game)
        else:  # Opponent's turn
            opponent.current_player = current_player_id # Ensure opponent knows ID
            action = opponent.select_action(game)
        
        # Check if the board is full
        if action is None:
            print(f"Game {game_num}: Draw (board full)")
            draws += 1
            done = True
            break
        
        # Make the move
        result = game.make_move(action)
        
        # If move failed, try again
        if not result:
            continue
        
        row, col = result
        
        # Print the board
        print(f"Game {game_num}, Move by Player {current_player_id}")
        game.print_board()
        print()
        
        # Check if the game is over
        if game.winning_moves(row, col):
            done = True
            if game.winner == 1:
                print(f"Game {game_num}: DQN agent won!")
                wins += 1
            else:
                print(f"Game {game_num}: Opponent won!")
                losses += 1
        
        # If the board is full after this move, it's a draw
        if not game.get_valid_columns() and not done:
            print(f"Game {game_num}: Draw (no valid moves)")
            draws += 1
            done = True
        
        # Switch to next player if game is not done
        if not done:
            game.current_player = (game.current_player % game.number_of_players) + 1

# Print final statistics
print("\nTest Results:")
print(f"Total Games: {NUM_TEST_GAMES}")
print(f"Wins: {wins} ({wins/NUM_TEST_GAMES:.2%})")
print(f"Losses: {losses} ({losses/NUM_TEST_GAMES:.2%})")
print(f"Draws: {draws} ({draws/NUM_TEST_GAMES:.2%})")