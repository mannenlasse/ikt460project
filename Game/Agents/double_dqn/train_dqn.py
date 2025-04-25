# Change these import lines
import numpy as np
import sys
import os
import torch
import argparse # Added for command-line arguments
import datetime # Import datetime module

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

# Opponent is always RandomAgent in this training script
opponent = RandomAgent(1)  # Will be assigned player_id 2 in the game loop

# Statistics tracking
win_history = []
loss_history = []
draw_history = []
reward_history = []
epsilon_history = []

# Get current timestamp for unique filenames
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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
    # --- End of inner game loop (while not done) ---

    # --- Add this section to record episode outcome ---
    if game.winner == dqn_agent.player_id:
        win_history.append(1)
        loss_history.append(0)
        draw_history.append(0)
    elif game.winner is not None: # Opponent won (winner is not None and not the agent)
        win_history.append(0)
        loss_history.append(1)
        draw_history.append(0)
    else: # Draw (winner is None and game is done)
        win_history.append(0)
        loss_history.append(0)
        draw_history.append(1)
    # --- End of section to add ---

    # Record other statistics (already existing, keep it)
    reward_history.append(total_reward)
    epsilon_history.append(dqn_agent.epsilon)

    # Print progress
    if (episode + 1) % 100 == 0:
        # Use max(1, ...) to avoid division by zero if less than 100 episodes have run
        # Ensure we don't index beyond list bounds if fewer than 100 episodes
        last_episodes_count = min(100, episode + 1)
        win_rate = sum(win_history[-last_episodes_count:]) / last_episodes_count
        loss_rate = sum(loss_history[-last_episodes_count:]) / last_episodes_count
        draw_rate = sum(draw_history[-last_episodes_count:]) / last_episodes_count
        avg_reward = sum(reward_history[-last_episodes_count:]) / last_episodes_count

        print(f"Episode {episode + 1}/{NUM_EPISODES}")
        print(f"Win Rate (last {last_episodes_count}): {win_rate:.2f}, Loss Rate: {loss_rate:.2f}, Draw Rate: {draw_rate:.2f}") # Clarified label
        print(f"Average Reward: {avg_reward:.4f}, Epsilon: {dqn_agent.epsilon:.4f}")
        print("-" * 50)

    # Save model periodically and at the end
    if (episode + 1) % SAVE_FREQUENCY == 0 or episode == NUM_EPISODES - 1:
        # Construct the full path using MODEL_DIR and add timestamp/reward_type/episode
        filename = f'dqn_agent_{REWARD_TYPE}_ep{episode + 1}_{timestamp}.pt'
        save_path = os.path.join(MODEL_DIR, filename)
        dqn_agent.save_model(save_path)

# Save the final model with more descriptive name
final_filename = f'dqn_agent_{REWARD_TYPE}_final_ep{NUM_EPISODES}_{timestamp}.pt'
final_save_path = os.path.join(MODEL_DIR, final_filename)
# We can just rename the last saved model if it was saved in the last step,
# or save it again explicitly. Saving again is simpler.
dqn_agent.save_model(final_save_path)
# Remove the old generic final save line if it exists:
# dqn_agent.save_model("models/dqn_agent_final.pt") # Remove or comment out this line

# --- Plotting ---
print("Training finished. Generating plots...")

# Plot training statistics
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
# Check if win_history is long enough before convolving
if len(win_history) >= 100:
    plt.plot(np.convolve(win_history, np.ones(100)/100, mode='valid'), 'g-', label='Smoothed Win Rate')
plt.plot(range(len(win_history)), win_history, 'g-', alpha=0.3, label='Raw Outcome (1=Win)') # Plot raw data too
plt.title('Win History') # Adjusted title
plt.xlabel('Episode')
plt.ylabel('Outcome')
plt.legend()


plt.subplot(2, 2, 2)
# Check if reward_history is long enough before convolving
if len(reward_history) >= 100:
    plt.plot(np.convolve(reward_history, np.ones(100)/100, mode='valid'), 'b-', label='Smoothed Reward')
plt.plot(range(len(reward_history)), reward_history, 'b-', alpha=0.3, label='Raw Reward')
plt.title('Reward History')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()


plt.subplot(2, 2, 3)
# Epsilon plot remains the same
plt.plot(range(len(epsilon_history)), epsilon_history, 'r-')
plt.title('Epsilon Decay')
plt.xlabel('Episode')
plt.ylabel('Epsilon')


# Modify the combined plot slightly for clarity
plt.subplot(2, 2, 4)
window_size = 100
# Calculate rolling averages safely, handle cases where i < window_size
win_rate_history_avg = [sum(win_history[max(0, i-window_size):i])/min(i, window_size) for i in range(1, len(win_history)+1)]
loss_rate_history_avg = [sum(loss_history[max(0, i-window_size):i])/min(i, window_size) for i in range(1, len(loss_history)+1)]
draw_rate_history_avg = [sum(draw_history[max(0, i-window_size):i])/min(i, window_size) for i in range(1, len(draw_history)+1)]

# Check if lists are non-empty before plotting
if win_rate_history_avg:
    plt.plot(range(len(win_rate_history_avg)), win_rate_history_avg, 'g-', label='Win Rate (Avg)')
if loss_rate_history_avg:
    plt.plot(range(len(loss_rate_history_avg)), loss_rate_history_avg, 'r-', label='Loss Rate (Avg)')
if draw_rate_history_avg:
    plt.plot(range(len(draw_rate_history_avg)), draw_rate_history_avg, 'b-', label='Draw Rate (Avg)')

plt.title(f'Performance Metrics (Rolling Avg, Window={window_size})')
plt.xlabel('Episode')
plt.ylabel('Rate')
# Only show legend if there's something to label
if win_rate_history_avg or loss_rate_history_avg or draw_rate_history_avg:
    plt.legend()


plt.tight_layout()
# Construct filename for the plot image including reward type and episodes
plot_filename = f'training_stats_{REWARD_TYPE}_ep{NUM_EPISODES}.png'
plot_save_path = os.path.join(os.path.dirname(__file__), plot_filename) # Save in the same directory as the script
plt.savefig(plot_save_path)
print(f"Plot saved to {plot_save_path}")
plt.show() # Ensure this line is NOT commented out if you want to see the plot window