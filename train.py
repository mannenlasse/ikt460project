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
from Game.Agents.double_dqn.double_dqn_agent import DoubleDQNAgent # Import DQN agent for opponent loading
from Game.reward_utils import calculate_reward # Import the centralized reward function
from Game.Agents.double_q_learning import QlearnAgent
# --- Central Model Directory ---
CENTRAL_MODEL_DIR = os.path.join(project_root, 'models')
os.makedirs(CENTRAL_MODEL_DIR, exist_ok=True)

# --- Plot Directory ---
PLOT_DIR = os.path.join(project_root, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)


# --- Game Parameters (but number of plyers is listed down below) ---
BOARD_HEIGHT = 6
BOARD_WIDTH = 7
WIN_LENGTH = 4

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Train an Agent for Connect Four.')
parser.add_argument('--model', type=str, required=True, choices=['dqn', 'qlearn', 'ppo'], help='Type of model/agent to train (e.g., dqn).') # Add more choices like 'ppo' later
parser.add_argument('--opponent', type=str, default='random', choices=['random', 'dqn_model'], help='Type of opponent agent (random or a loaded dqn_model).')
parser.add_argument('--opponent_model_path', type=str, default=None, help='Path to the pre-trained opponent model file (.pt) if opponent is dqn_model.')
parser.add_argument('--reward_type', type=str, default='sparse', choices=['sparse', 'shaped'], help='Type of reward structure.')
parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train.')
parser.add_argument('--save_freq', type=int, default=5000, help='Frequency (in episodes) to save the model.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')

# Add other relevant hyperparameters as needed (e.g., batch_size, memory_size for DQN)
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (if applicable).')
parser.add_argument('--memory_size', type=int, default=50000, help='Replay memory size (if applicable).')


args = parser.parse_args()

# Assign args to variables
MODEL_TYPE = args.model
OPPONENT_TYPE = args.opponent
OPPONENT_MODEL_PATH = args.opponent_model_path 
REWARD_TYPE = args.reward_type
NUM_EPISODES = args.episodes
SAVE_FREQUENCY = args.save_freq
LEARNING_RATE = args.lr
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

elif MODEL_TYPE == 'qlearn':
    print(f"Initializing Double Q-Learning Agent...")
    agent = QlearnAgent(
        player_id=1,
        learn_rate=LEARNING_RATE,
        disc_factor=0.99,  # can add --gamma to CLI if needed
        explor_rate=1.0,
        explor_decay=0.995
    )

# Add elif blocks here for other models like PPO later
# elif MODEL_TYPE == 'ppo':
#     from Game.Agents.ppo.ppo_agent import PPOAgent # Example
#     agent = PPOAgent(...)
else:
    print(f"Error: Unknown model type '{MODEL_TYPE}'")
    sys.exit(1)



# --- Opponent Initialization ---
# Define the full list of opponents (can be mixed: random or dqn with model path)


#hva skjer: this functions maps a player with an id. The agent who is learning is always player 1 while the others players are oppoents and you can chose what kind 
def build_player_map(agent, opponent_defs, board_height, board_width, board_column):
    player_map = {1: agent}  # Learning agent always player 1

    for i, (kind, model_path) in enumerate(opponent_defs):
        playerid = i + 2  # Opponent player IDs start at 2

        if kind == "random":
            player_map[playerid] = RandomAgent(Current_Player=playerid)

        elif kind == "dqn":
            dqn = DoubleDQNAgent(
                player_id=playerid,
                board_height=board_height,
                board_width=board_width,
                action_size=board_column,
                learning_rate=0.0,
                gamma=0.99,
                epsilon=0.0,
                epsilon_min=0.0,
                epsilon_decay=1.0
            )
            if model_path:
                dqn.load_model(model_path)
                dqn.model.eval()
                if hasattr(dqn, "target_model"):
                    dqn.target_model.eval()
            player_map[playerid] = dqn

        else:
            raise ValueError(f"Unknown agent type: {kind}")

    for playerid, agent_obj in sorted(player_map.items()):
        print(f"   Player {playerid}: {type(agent_obj).__name__}", flush=True)

    return player_map




#HERE YOU CHOOSE WHAT KIND OF PLAYERS YOU WANT, SO RIGHT NOW ITS 67 RANDOM PLAYERS
opponent_definitions = [("random", None)]   # Fill to 70 players total (69 + agent = 70)

NUM_PLAYERS = len(opponent_definitions) + 1

player_map = build_player_map(agent, opponent_definitions, BOARD_HEIGHT, BOARD_WIDTH, BOARD_WIDTH)
for playerId, agent_obj in player_map.items():
    print(f"  Player {playerId}: {type(agent_obj).__name__}")






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
epsilon_history = [] 


# Get current timestamp for unique filenames
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")



# --- Training Loop ---
print(f"\n--- Starting Training ---")
print(f"Model: {MODEL_TYPE}, Opponent: {OPPONENT_TYPE}, Reward: {REWARD_TYPE}")
print(f"Episodes: {NUM_EPISODES}, LR: {LEARNING_RATE}")
print(f"-------------------------\n")

for episode in range(NUM_EPISODES):
    #initiate game 
    game = Game(BOARD_HEIGHT, BOARD_WIDTH, NUM_PLAYERS, WIN_LENGTH)
    done = False
    #reward for one episode to log how good the agent does per game
    total_episode_reward = 0
    state = None 

    while not done:
        current_player_id = game.current_player
        current_agent = None
        is_learning_agent_turn = False

        current_agent = player_map[current_player_id]

        is_learning_agent_turn = (current_player_id == agent.player_id)

        action = current_agent.select_action(game)

        # Handle board full before attemtping a move
        if action is None:
            done = True
            # If it was the learning agent's turn to move but couldn't, store the final transition.
            if is_learning_agent_turn and state is not None and hasattr(agent, 'remember'):
                 # Get the final state or at least the current state
                 next_state = agent.get_state(game) if hasattr(agent, 'get_state') else state
                
                 # calcuate reward during last state to learn
                 final_reward = calculate_reward(game, agent.player_id, -1, -1, True, REWARD_TYPE) # Use dummy row/col
 
                #store what has happened so that the agent learns
                 agent.remember(state, -1, final_reward, next_state, True) 

                #track total reward
                 total_episode_reward += final_reward 
            break 
        else:
            #Making the move
            try:
                #dropping the piece in a column and retuning row and column
                row, col = game.make_move(action) #
            #check error with the index(so invalid drops) 
            except Exception as e:
                print(f"Error during make_move for action {action}: {e}")
                pass

            # checks if the move creates a win condition
            if game.winning_moves(row, col): 
                done = True

            # if the board is full regardless, end the game
            elif not game.get_valid_columns(): 
                done = True


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

            # storing expereince n updates for learning agent
            if is_learning_agent_turn:
                total_episode_reward += reward  

                if state is not None:
                    if MODEL_TYPE == 'qlearn':
                        agent.observe(reward, game, done)
                    if MODEL_TYPE == 'dqn':
                        agent.remember(state, action, reward, next_state, done)

            #  Train Agent after storing experience
            if is_learning_agent_turn and hasattr(agent, 'train'):
                 agent.train()

            # Switch player if game not done 
            if not done:
                game.current_player = (game.current_player % NUM_PLAYERS) + 1


    # --- End of Episode ---

    # Decay epsilon ONCE per episode (if agent has epsilon)
    if hasattr(agent, 'epsilon') and hasattr(agent, 'epsilon_min') and hasattr(agent, 'epsilon_decay'):
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

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

    # Add this function near the top of the file, after the imports but before the training loop
    def get_next_version(base_filename):
        """Find the next available version number for a file"""
        i = 1
        while True:
            if i == 1:
                test_filename = base_filename
            else:
                # Insert version number before the file extension
                name_parts = base_filename.rsplit('.', 1)
                test_filename = f"{name_parts[0]}_v{i}.{name_parts[1]}"
            
            if not os.path.exists(test_filename):
                return test_filename
            i += 1
    
    # Then remove the function definition from inside the model saving block
    # In the model saving section, change:
    if hasattr(agent, 'save_model') and ((episode + 1) % SAVE_FREQUENCY == 0 or episode == NUM_EPISODES - 1):
        # Lag en kompakt beskrivelse av motstandere for filnavnet
        from collections import Counter
        opponent_types = [kind for kind, _ in opponent_definitions]
        opponent_counts = Counter(opponent_types)
        
        opponent_filename_parts = []
        for kind, count in opponent_counts.items():
            opponent_filename_parts.append(f"{count}x{kind}")
        
        opponents_str = "_".join(opponent_filename_parts)
        
        # Inkluder brettdimensjoner og win_length i filnavnet
        board_info = f"{BOARD_HEIGHT}x{BOARD_WIDTH}"
        win_info = f"win{WIN_LENGTH}"
        
        # Konstruer filnavn: modelType_rewardType_vs_opponents_boardSize_winLength_epN.pt
        if MODEL_TYPE == "qlearn":
            ext = ".pkl"
        else:
            ext = ".pt"
        
        filename = f'{MODEL_TYPE}_{REWARD_TYPE}_vs_{opponents_str}_{board_info}_{win_info}_ep{episode + 1}{ext}'
        save_path = os.path.join(CENTRAL_MODEL_DIR, filename)
        save_path = get_next_version(save_path)  # Use the function here
        
        # Debug-utskrift for å verifisere at lagringsfunksjonen blir kalt
        print(f"Attempting to save model to: {save_path}")
        
        # Sørg for at mappen eksisterer
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            agent.save_model(save_path)
            print(f"--- Model successfully saved at episode {episode + 1} to {save_path} ---")
        except Exception as e:
            print(f"Error saving model: {e}")


# Optionally display the plot
# plt.show()

print("\n--- Training Complete ---")

# After the training loop is complete, generate the plots
# --- Plotting Results ---
plt.figure(figsize=(15, 10)) # Adjusted figure size for 4 plots

# --- Data Preparation for Plots ---
window_size = 100 # Moving average window
episodes_axis = np.arange(1, NUM_EPISODES + 1)

def moving_average(data, window):
    if len(data) == 0:
        return np.array([])
    actual_window = min(len(data), window)
    if actual_window == 0: 
        return np.array([])
    return np.convolve(data, np.ones(actual_window)/actual_window, mode='valid')

# Ensure we have data to plot
if len(win_history) > 0:
    ma_win_rate = moving_average(win_history, window_size)
    ma_reward = moving_average(reward_history, window_size)

    # Calculate the correct episodes for moving average
    ma_episodes = episodes_axis[window_size-1:] if len(ma_win_rate) > 0 else np.array([])

    # --- Plot 1: Win Rate (Top-Left) ---
    plt.subplot(2, 2, 1)
    if len(ma_episodes) > 0:
        plt.plot(ma_episodes, ma_win_rate, label=f'Win Rate (MA {window_size})', color='green')
        plt.plot(episodes_axis, win_history, alpha=0.3, color='lightgreen')
    else:
        plt.plot(episodes_axis, win_history, label='Win Rate', color='green')
    plt.title('Win Rate')
    plt.xlabel('Episodes')
    plt.ylabel('Rate')
    plt.ylim(-0.1, 1.1)  # Set y-axis limits for win rate
    plt.legend()
    plt.grid(True)

    # --- Plot 2: Average Reward (Top-Right) ---
    plt.subplot(2, 2, 2)
    if len(ma_episodes) > 0:
        plt.plot(ma_episodes, ma_reward, label=f'Avg Reward (MA {window_size})', color='purple')
        plt.plot(episodes_axis, reward_history, alpha=0.3, color='plum')
    else:
        plt.plot(episodes_axis, reward_history, label='Avg Reward', color='purple')
    plt.title('Average Reward')
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
    plt.ylim(-0.1, 1.1)  # Set y-axis limits for epsilon
    plt.legend()
    plt.grid(True)

    # --- Plot 4: Learning Progress (Bottom-Right) ---
    plt.subplot(2, 2, 4)
    if len(win_history) > window_size:
        # Calculate win rate improvement over time
        window = 100
        improvement = []
        for i in range(window, len(win_history), window):
            prev_win_rate = np.mean(win_history[i-window:i])
            curr_win_rate = np.mean(win_history[i:min(i+window, len(win_history))])
            improvement.append(curr_win_rate - prev_win_rate)
        
        # Plot improvement
        x = np.arange(window, len(win_history), window)[:len(improvement)]
        plt.bar(x, improvement, width=window*0.8, color='purple')
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # Find the max absolute value for symmetric y-axis limits
        max_abs_change = max(abs(min(improvement)), abs(max(improvement))) if improvement else 0.2
        # Add 20% padding and round to nearest 0.05
        y_limit = min(0.3, round(max_abs_change * 1.2 * 20) / 20)
        plt.ylim(-y_limit, y_limit)  # Set symmetric limits around zero
        
        # Updated title and y-axis label
        plt.title('Learning Progress')
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate Change')
        plt.grid(True, axis='y')
    else:
        plt.text(0.5, 0.5, 'Not enough data for learning progress', ha='center', va='center')
        plt.title('Learning Progress')
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate Change')
    

else:
    # If no data, display a message
    plt.text(0.5, 0.5, 'No training data available', ha='center', va='center', transform=plt.gcf().transFigure)

# Create a description of opponents for the plot title
from collections import Counter
opponent_types = [kind for kind, _ in opponent_definitions]
opponent_counts = Counter(opponent_types)

# For the plot title
opponent_desc_list = []
for kind, count in opponent_counts.items():
    if kind == "random":
        opponent_desc_list.append(f"{count}×Random")
    elif kind == "dqn":
        opponent_desc_list.append(f"{count}×DQN")
    elif kind == "ppo":
        opponent_desc_list.append(f"{count}×PPO")

# Join opponent descriptions
opponents_title = ", ".join(opponent_desc_list)

# Add board dimensions to title
board_info = f"{BOARD_HEIGHT}x{BOARD_WIDTH}"

main_plot_title = f'Training Performance: {MODEL_TYPE.upper()} ({REWARD_TYPE}) vs {opponents_title} ({board_info}, {NUM_EPISODES} Episodes)'
plt.suptitle(main_plot_title, fontsize=16)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# For the filename
opponent_filename_parts = []
for kind, count in opponent_counts.items():
    opponent_filename_parts.append(f"{count}x{kind}")

opponents_filename = "_".join(opponent_filename_parts)
board_info = f"{BOARD_HEIGHT}x{BOARD_WIDTH}"
win_info = f"win{WIN_LENGTH}"
plot_filename = f'plot_{MODEL_TYPE}_{REWARD_TYPE}_vs_{opponents_filename}_{board_info}_{win_info}_ep{NUM_EPISODES}.png'

# Make sure the plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

plot_save_path = os.path.join(PLOT_DIR, plot_filename)
plot_save_path = get_next_version(plot_save_path)
plt.savefig(plot_save_path)
print(f"Plot saved to: {plot_save_path}")
