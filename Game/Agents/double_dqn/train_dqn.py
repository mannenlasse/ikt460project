# Change these import lines
import numpy as np
import sys
import os
import torch
import argparse # Added for command-line arguments

# Add the project root to the path so imports work correctly
# Corrected path calculation: go up three levels from __file__
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)


from Game.game import Game
from Game.Agents.random_agent import RandomAgent
# Change relative import to absolute import
from Game.Agents.double_dqn.double_dqn_agent import DoubleDQNAgent
import matplotlib.pyplot as plt
# Removed duplicate os import

# Create directory for saving models relative to this script
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Game parameters
BOARD_HEIGHT = 6
BOARD_WIDTH = 7
NUM_PLAYERS = 2
WIN_LENGTH = 4
NUM_EPISODES = 10000
SAVE_FREQUENCY = 500

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Train a Double DQN Agent for Connect Four.')
parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train.')
parser.add_argument('--reward_type', type=str, default='sparse', choices=['sparse', 'shaped'], help='Type of reward structure to use.')
parser.add_argument('--save_freq', type=int, default=500, help='Frequency (in episodes) to save the model.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
# Add other hyperparameters if needed
args = parser.parse_args()

NUM_EPISODES = args.episodes
REWARD_TYPE = args.reward_type
SAVE_FREQUENCY = args.save_freq
LEARNING_RATE = args.lr
# --- End Argument Parsing ---


# Initialize agents
print(f"Initializing DQN Agent with {REWARD_TYPE} rewards...")
dqn_agent = DoubleDQNAgent(
    player_id=1, # DQN agent always starts as player 1 in training
    board_height=BOARD_HEIGHT,
    board_width=BOARD_WIDTH,
    action_size=BOARD_WIDTH,
    memory_size=50000,
    batch_size=64,
    learning_rate=LEARNING_RATE,
    reward_type=REWARD_TYPE # Pass the reward type
)

# Legg til GPU-akselerasjon hvis tilgjengelig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dqn_agent.model.to(device)
dqn_agent.target_model.to(device)

# Oppdater også select_action og train metodene i DoubleDQNAgent-klassen for å bruke device
# Dette kan gjøres ved å modifisere double_dqn_agent.py

opponent = RandomAgent(1)  # Will be assigned player_id 2 in the game loop

# Statistics tracking
win_history = []
loss_history = []
draw_history = []
reward_history = []
epsilon_history = []

# Training loop
for episode in range(NUM_EPISODES):
    # Create a new game
    game = Game(BOARD_HEIGHT, BOARD_WIDTH, NUM_PLAYERS, WIN_LENGTH)
    
    # Reset episode variables
    done = False
    total_reward = 0
    
    # Store game history for training
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    
    # Game loop
    while not done:
        # Determine current player and agent
        current_player_id = game.current_player
        if current_player_id == dqn_agent.player_id:
            current_agent = dqn_agent
            # Get state only when it's DQN's turn to act
            state = dqn_agent.get_state(game)
        else:
            current_agent = opponent
            opponent.current_player = current_player_id # Ensure opponent knows its ID

        # Select action
        action = current_agent.select_action(game)

        # Handle board full scenario
        if action is None:
            done = True
            reward = 0 # No reward change on draw by board full
            # Store final experience if it was DQN's turn to move but couldn't
            if current_player_id == dqn_agent.player_id and state is not None:
                 next_state = dqn_agent.get_state(game) # Get final state
                 # Use calculate_reward for consistency, even though it's likely 0 for draw
                 final_reward = dqn_agent.calculate_reward(game, -1, -1, True) # Use dummy row/col
                 dqn_agent.remember(state, action, final_reward, next_state, True)
                 total_reward += final_reward
            break # Exit while loop

        # Make the move
        result = game.make_move(action)
        if not result:
            # This should ideally not happen if select_action respects valid moves
            print(f"Warning: Agent {current_player_id} chose invalid move {action}. Skipping turn.")
            # Switch player manually if move failed, or handle differently
            game.current_player = (game.current_player % NUM_PLAYERS) + 1
            continue

        row, col = result
        next_state_for_memory = dqn_agent.get_state(game) # Get next state after move

        # Check if game is over
        if game.winning_moves(row, col):
            done = True
        elif not game.get_valid_columns(): # Check for draw after checking win
            done = True

        # Calculate reward using the agent's method
        # This happens *after* the move and win/draw check
        reward = dqn_agent.calculate_reward(game, row, col, done)

        # Store experience for the DQN agent only if it was its turn
        if current_player_id == dqn_agent.player_id:
            dqn_agent.remember(state, action, reward, next_state_for_memory, done)
            total_reward += reward

        # Train the agent
        if current_player_id == dqn_agent.player_id: # Train only after DQN agent's move
             dqn_agent.train() # Train based on experiences in memory

        # Switch player only if game not done
        if not done:
            game.current_player = (game.current_player % NUM_PLAYERS) + 1

    # Record statistics
    reward_history.append(total_reward)
    epsilon_history.append(dqn_agent.epsilon)
    
    # Print progress
    if (episode + 1) % 100 == 0:
        win_rate = sum(win_history[-100:]) / 100
        loss_rate = sum(loss_history[-100:]) / 100
        draw_rate = sum(draw_history[-100:]) / 100
        avg_reward = sum(reward_history[-100:]) / 100
        
        print(f"Episode {episode + 1}/{NUM_EPISODES}")
        print(f"Win Rate: {win_rate:.2f}, Loss Rate: {loss_rate:.2f}, Draw Rate: {draw_rate:.2f}")
        print(f"Average Reward: {avg_reward:.4f}, Epsilon: {dqn_agent.epsilon:.4f}")
        print("-" * 50)
    
    # Save model periodically and at the end
    if (episode + 1) % SAVE_FREQUENCY == 0 or episode == NUM_EPISODES - 1:
        save_path = os.path.join(MODEL_DIR, f'dqn_agent_{REWARD_TYPE}_ep{episode + 1}.pt')
        dqn_agent.save_model(save_path)

# Save the final model
dqn_agent.save_model("models/dqn_agent_final.pt")

# Plot training statistics
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(range(len(win_history)), win_history, 'g-', alpha=0.3)
plt.plot(np.convolve(win_history, np.ones(100)/100, mode='valid'), 'g-')
plt.title('Win Rate')
plt.xlabel('Episode')
plt.ylabel('Win (1) / Loss (0)')

plt.subplot(2, 2, 2)
plt.plot(range(len(reward_history)), reward_history, 'b-', alpha=0.3)
plt.plot(np.convolve(reward_history, np.ones(100)/100, mode='valid'), 'b-')
plt.title('Reward History')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(2, 2, 3)
plt.plot(range(len(epsilon_history)), epsilon_history, 'r-')
plt.title('Epsilon Decay')
plt.xlabel('Episode')
plt.ylabel('Epsilon')

plt.subplot(2, 2, 4)
window_size = 100
win_rate_history = [sum(win_history[max(0, i-window_size):i])/min(i, window_size) for i in range(1, len(win_history)+1)]
loss_rate_history = [sum(loss_history[max(0, i-window_size):i])/min(i, window_size) for i in range(1, len(loss_history)+1)]
draw_rate_history = [sum(draw_history[max(0, i-window_size):i])/min(i, window_size) for i in range(1, len(draw_history)+1)]

plt.plot(range(len(win_rate_history)), win_rate_history, 'g-', label='Win Rate')
plt.plot(range(len(loss_rate_history)), loss_rate_history, 'r-', label='Loss Rate')
plt.plot(range(len(draw_rate_history)), draw_rate_history, 'b-', label='Draw Rate')
plt.title('Performance Metrics')
plt.xlabel('Episode')
plt.ylabel('Rate')
plt.legend()

plt.tight_layout()
plt.savefig('training_stats.png')
plt.show()