import numpy as np
import sys
import os
import torch
import argparse
import datetime
import matplotlib.pyplot as plt

# Add the project root to the path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import necessary components
from Game.game import Game
from Game.Agents.random_agent import RandomAgent
# Import agent classes dynamically later based on args
# from Game.Agents.double_dqn.double_dqn_agent import DoubleDQNAgent # Example
from Game.reward_utils import calculate_reward # Import the centralized reward function

# --- Central Model Directory ---
CENTRAL_MODEL_DIR = os.path.join(project_root, 'models')
os.makedirs(CENTRAL_MODEL_DIR, exist_ok=True)

# --- Plot Directory ---
PLOT_DIR = os.path.join(project_root, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)


# --- Game Parameters (Defaults) ---
BOARD_HEIGHT = 6
BOARD_WIDTH = 7
NUM_PLAYERS = 2
WIN_LENGTH = 4

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Train an Agent for Connect Four.')
parser.add_argument('--model', type=str, required=True, choices=['dqn'], help='Type of model/agent to train (e.g., dqn).') # Add more choices like 'ppo' later
parser.add_argument('--opponent', type=str, default='random', choices=['random'], help='Type of opponent agent.') # Add more choices later
parser.add_argument('--reward_type', type=str, default='sparse', choices=['sparse', 'shaped'], help='Type of reward structure.')
parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train.')
parser.add_argument('--save_freq', type=int, default=5000, help='Frequency (in episodes) to save the model.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
# Add other relevant hyperparameters as needed (e.g., batch_size, memory_size for DQN)
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (if applicable).')
parser.add_argument('--memory_size', type=int, default=50000, help='Replay memory size (if applicable).')

args = parser.parse_args()

# Assign args to variables
NUM_EPISODES = args.episodes
REWARD_TYPE = args.reward_type
SAVE_FREQUENCY = args.save_freq
LEARNING_RATE = args.lr
MODEL_TYPE = args.model
OPPONENT_TYPE = args.opponent
BATCH_SIZE = args.batch_size
MEMORY_SIZE = args.memory_size

# --- Agent Initialization ---
agent = None
if MODEL_TYPE == 'dqn':
    from Game.Agents.double_dqn.double_dqn_agent import DoubleDQNAgent
    print(f"Initializing Double DQN Agent...")
    agent = DoubleDQNAgent(
        player_id=1, # Agent always starts as player 1
        board_height=BOARD_HEIGHT,
        board_width=BOARD_WIDTH,
        action_size=BOARD_WIDTH,
        memory_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
        # reward_type is handled externally now
    )
# Add elif blocks here for other models like PPO later
# elif MODEL_TYPE == 'ppo':
#     from Game.Agents.ppo.ppo_agent import PPOAgent # Example
#     agent = PPOAgent(...)
else:
    print(f"Error: Unknown model type '{MODEL_TYPE}'")
    sys.exit(1)

# --- Opponent Initialization ---
opponent = None
if OPPONENT_TYPE == 'random':
    print(f"Initializing Random Opponent...")
    # opponent = RandomAgent() # Initialize without player_id argument # <-- Incorrect line from previous fix
    opponent = RandomAgent(Current_Player=2) # Use the correct argument name 'Current_Player' and assign ID 2
else:
    # Add elif for other opponents later
    print(f"Error: Unknown opponent type '{OPPONENT_TYPE}'")
    sys.exit(1)


# --- GPU Acceleration (If agent supports it) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Move agent's models to device if it's a PyTorch model
if hasattr(agent, 'model') and hasattr(agent.model, 'to'):
    agent.model.to(device)
if hasattr(agent, 'target_model') and hasattr(agent.target_model, 'to'): # For DQN
    agent.target_model.to(device)
# Add similar checks/calls for other frameworks or model structures if needed

# --- Statistics Tracking ---
win_history = []
loss_history = []
draw_history = []
reward_history = []
epsilon_history = [] # Specific to agents with epsilon (like DQN)

# Get current timestamp for unique filenames
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Training Loop ---
print(f"\n--- Starting Training ---")
print(f"Model: {MODEL_TYPE}, Opponent: {OPPONENT_TYPE}, Reward: {REWARD_TYPE}")
print(f"Episodes: {NUM_EPISODES}, LR: {LEARNING_RATE}")
print(f"-------------------------\n")

for episode in range(NUM_EPISODES):
    game = Game(BOARD_HEIGHT, BOARD_WIDTH, NUM_PLAYERS, WIN_LENGTH)
    done = False
    total_episode_reward = 0
    state = None # Initialize state

    while not done:
        current_player_id = game.current_player
        current_agent = None
        is_learning_agent_turn = False

        if current_player_id == agent.player_id:
            current_agent = agent
            is_learning_agent_turn = True
            # Get state representation needed by the learning agent
            if hasattr(agent, 'get_state'):
                 state = agent.get_state(game)
            else:
                 # Handle state representation for other agent types if needed
                 state = game.board.flatten() # Default fallback?
        else:
            current_agent = opponent
            # Ensure opponent knows its ID if needed by its select_action
            if hasattr(opponent, 'current_player'):
                 opponent.current_player = current_player_id

        # Select action
        action = current_agent.select_action(game)

        # Handle board full before move attempt (if select_action returns None)
        if action is None:
            done = True
            # If it was the learning agent's turn to move but couldn't, store the final transition.
            if is_learning_agent_turn and state is not None and hasattr(agent, 'remember'):
                 # Get the final state
                 next_state = agent.get_state(game) if hasattr(agent, 'get_state') else state
                 # Calculate reward for the agent's perspective for this final state (draw = 0 usually)
                 final_reward = calculate_reward(game, agent.player_id, -1, -1, True, REWARD_TYPE) # Use dummy row/col
                 # Store experience. The agent's remember method needs to handle cases where action might be None
                 # or we need a placeholder. DoubleDQNAgent expects an int action. Let's pass -1 as a placeholder.
                 # Ensure your DoubleDQNAgent.remember can handle action=-1 or modify if needed.
                 # For now, assuming -1 is acceptable or handled.
                 agent.remember(state, -1, final_reward, next_state, True) # Use action = -1 placeholder
                 total_episode_reward += final_reward # Add final reward if any
            # No move was made, break the inner loop
            break # Exit the while loop immediately after handling the no-action state

        else:
            # --- Make the move ---
            # Ensure make_move returns row, col or handles errors appropriately
            # Assuming make_move returns (row, col) on success and raises error or returns None/False on failure
            try:
                row, col = game.make_move(action) # Get row and col where piece landed
            except Exception as e: # Or check return value if make_move indicates failure differently
                print(f"Error during make_move for action {action}: {e}")
                # Decide how to handle invalid move attempt (e.g., skip turn, end game?)
                # For now, let's assume make_move is robust or select_action only gives valid moves.
                # If make_move could fail gracefully, add handling here.
                # Let's assume it works if action was valid.
                pass # Continue assuming make_move succeeded if no exception

            # --- Check if game is over (using methods from train_dqn.py) ---
            # Check win first
            if game.winning_moves(row, col): # Use winning_moves like in train_dqn.py
                done = True
            # Check draw only if not won
            elif not game.get_valid_columns(): # Use get_valid_columns like in train_dqn.py
                done = True
            # else: done remains False

            # --- Calculate Reward ---
            # Reward is calculated *after* the move and win/draw check, using the 'done' status
            reward = calculate_reward(game, agent.player_id, row, col, done, REWARD_TYPE)

            # --- Get Next State ---
            next_state = None
            if hasattr(agent, 'get_state'):
                next_state = agent.get_state(game)
            else:
                # Default fallback if agent doesn't have a specific get_state
                next_state = game.board.flatten()

            # --- Store Experience (if it was the learning agent's turn) ---
            if is_learning_agent_turn and state is not None:
                if hasattr(agent, 'remember'):
                    # Pass the state *before* the action, the action taken, the calculated reward,
                    # the state *after* the action, and the final done status.
                    agent.remember(state, action, reward, next_state, done)
                total_episode_reward += reward # Accumulate reward received by the agent

            # --- Train Agent (if applicable) ---
            if is_learning_agent_turn and hasattr(agent, 'train'):
                 # Train after storing experience
                 agent.train()

            # --- Switch player only if game not done ---
            if not done:
                game.current_player = (game.current_player % NUM_PLAYERS) + 1

    # --- End of Episode ---

    # Record episode outcome
    if game.winner == agent.player_id:
        win_history.append(1)
        loss_history.append(0)
        draw_history.append(0)
    elif game.winner is not None: # Opponent won
        win_history.append(0)
        loss_history.append(1)
        draw_history.append(0)
    else: # Draw
        win_history.append(0)
        loss_history.append(0)
        draw_history.append(1)

    # Record other statistics
    reward_history.append(total_episode_reward)
    if hasattr(agent, 'epsilon'): # Only record epsilon if the agent has it
        epsilon_history.append(agent.epsilon)

    # Print progress
    if (episode + 1) % 100 == 0 or episode == NUM_EPISODES - 1:
        last_episodes_count = min(100, episode + 1)
        win_rate = sum(win_history[-last_episodes_count:]) / last_episodes_count
        loss_rate = sum(loss_history[-last_episodes_count:]) / last_episodes_count
        draw_rate = sum(draw_history[-last_episodes_count:]) / last_episodes_count
        avg_reward = sum(reward_history[-last_episodes_count:]) / last_episodes_count

        print(f"Episode {episode + 1}/{NUM_EPISODES}")
        print(f"  Win Rate (last {last_episodes_count}): {win_rate:.2f}, Loss Rate: {loss_rate:.2f}, Draw Rate: {draw_rate:.2f}")
        print(f"  Avg Reward: {avg_reward:.4f}", end="")
        if epsilon_history: # Print epsilon if available
             print(f", Epsilon: {epsilon_history[-1]:.4f}")
        else:
             print() # Newline if no epsilon
        print("-" * 50)

    # Save model periodically and at the end
    if hasattr(agent, 'save_model') and ( (episode + 1) % SAVE_FREQUENCY == 0 or episode == NUM_EPISODES - 1 ):
        # Construct filename: modelType_rewardType_vs_opponentType_ep<N>_timestamp.pt
        filename = f'{MODEL_TYPE}_{REWARD_TYPE}_vs_{OPPONENT_TYPE}_ep{episode + 1}_{timestamp}.pt'
        save_path = os.path.join(CENTRAL_MODEL_DIR, filename)
        agent.save_model(save_path)
        # Add the confirmation print statement here
        print(f"--- Model saved at episode {episode + 1} to {save_path} ---")

# --- Plotting Results ---
# Corrected from sprint to print
print("\n--- Generating Plots ---")
plt.figure(figsize=(15, 10)) # Adjusted figure size for 4 plots

# --- Data Preparation for Plots ---
window_size = 100 # Moving average window
episodes_axis = np.arange(1, NUM_EPISODES + 1)

def moving_average(data, window):
    actual_window = min(len(data), window)
    if actual_window == 0: return np.array([])
    return np.convolve(data, np.ones(actual_window)/actual_window, mode='valid')

ma_win_rate = moving_average(win_history, window_size)
ma_loss_rate = moving_average(loss_history, window_size) # Still needed for comparison if desired, or remove if not plotting MA loss
ma_draw_rate = moving_average(draw_history, window_size) # Still needed for comparison if desired, or remove if not plotting MA draw
ma_reward = moving_average(reward_history, window_size)

actual_window_used = min(len(episodes_axis), window_size)
ma_episodes = episodes_axis[actual_window_used - 1:] if actual_window_used > 0 else np.array([])

# Ensure lengths match for MA plots (simple safeguard)
if len(ma_episodes) != len(ma_win_rate) and len(ma_win_rate) > 0:
    ma_episodes = episodes_axis[len(episodes_axis)-len(ma_win_rate):]
elif len(ma_win_rate) == 0:
     ma_episodes = np.array([])

# --- Plot 1: Smoothed Win Rate (Top-Left) ---
plt.subplot(2, 2, 1)
if len(ma_episodes) > 0:
    plt.plot(ma_episodes, ma_win_rate, label=f'Win Rate (MA {window_size})', color='green')
    # Optionally add loss/draw MA here too if desired
    # plt.plot(ma_episodes, ma_loss_rate, label=f'Loss Rate (MA {window_size})', color='red', alpha=0.6)
    # plt.plot(ma_episodes, ma_draw_rate, label=f'Draw Rate (MA {window_size})', color='blue', alpha=0.6)
else:
    plt.text(0.5, 0.5, 'Not enough data for MA', ha='center', va='center')
plt.title('Smoothed Win Rate')
plt.xlabel('Episodes')
plt.ylabel('Rate')
plt.legend()
plt.grid(True)

# --- Plot 2: Smoothed Average Reward (Top-Right) ---
plt.subplot(2, 2, 2)
if len(ma_episodes) > 0:
    plt.plot(ma_episodes, ma_reward, label=f'Avg Reward (MA {window_size})', color='purple')
else:
    plt.text(0.5, 0.5, 'Not enough data for MA', ha='center', va='center')
plt.title('Smoothed Average Reward')
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)

# --- Plot 3: Epsilon Decay (Bottom-Left) ---
plt.subplot(2, 2, 3)
if epsilon_history and len(episodes_axis) > 0:
    plt.plot(episodes_axis, epsilon_history, label='Epsilon', color='red', linestyle=':')
    plt.ylabel('Epsilon')
else:
     plt.text(0.5, 0.5, 'No Epsilon data', ha='center', va='center')
plt.title('Epsilon Decay')
plt.xlabel('Episodes')
plt.legend()
plt.grid(True)

# --- Plot 4: Raw Win/Loss/Draw Rates (Bottom-Right) ---
plt.subplot(2, 2, 4)
if len(episodes_axis) > 0:
    plt.plot(episodes_axis, win_history, label='Win Rate (Raw)', color='green', alpha=0.7)
    plt.plot(episodes_axis, loss_history, label='Loss Rate (Raw)', color='red', alpha=0.7)
    plt.plot(episodes_axis, draw_history, label='Draw Rate (Raw)', color='blue', alpha=0.7)
else:
     plt.text(0.5, 0.5, 'No data', ha='center', va='center')
plt.title('Raw Performance Metrics')
plt.xlabel('Episodes')
plt.ylabel('Rate (0 or 1)')
plt.legend()
plt.grid(True)


# --- Final Touches ---
# Add a main title for the whole figure
main_plot_title = f'Training Performance: {MODEL_TYPE.upper()} ({REWARD_TYPE}) vs {OPPONENT_TYPE.capitalize()} ({NUM_EPISODES} Episodes)'
plt.suptitle(main_plot_title, fontsize=16) # Use suptitle for figure-level title

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap and make space for suptitle

# Save the plot
plot_filename = f'plot_{MODEL_TYPE}_{REWARD_TYPE}_vs_{OPPONENT_TYPE}_ep{NUM_EPISODES}_{timestamp}.png'
plot_save_path = os.path.join(PLOT_DIR, plot_filename) # Save in PLOT_DIR
plt.savefig(plot_save_path)
print(f"Plot saved to: {plot_save_path}")

# Optionally display the plot
# plt.show()

print("\n--- Training Complete ---")