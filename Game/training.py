import argparse
import numpy as np
import os
from game import Game
from reward_utils import calculate_reward
import subprocess
from Agents.random_agent import RandomAgent
from Agents.ppo_agent import PPOAgent
from Agents.double_dqn_agent import DoubleDQNAgent
from Agents.double_q_learning import QlearnAgent
import json
from datetime import datetime
import copy # Added for deepcopying model state

BOARD_HEIGHT = 6
BOARD_WIDTH = 7
WIN_LENGTH = 4
NUM_EPISODES = 30000

AGENT_CLASSES = {
    "qlearn": QlearnAgent,
    "dqn": DoubleDQNAgent,
    "ppo": PPOAgent,
    "random": RandomAgent
}

def init_agents(agent_names):
    agents = []
    for i, name in enumerate(agent_names):
        player_id = i + 1
        if name == "qlearn":
            agent = QlearnAgent(learn_rate=0.05, disc_factor=0.95, explor_rate=1.0, explor_decay=0.997, player_id=player_id
)
        elif name == "dqn":
            agent = DoubleDQNAgent(board_height=BOARD_HEIGHT,board_width=BOARD_WIDTH,action_size=BOARD_WIDTH,learning_rate=0.0005,   gamma=0.99,  
                                     epsilon=1.0,epsilon_min=0.05,epsilon_decay=0.999,  player_id=player_id)
        elif name == "ppo":
            agent = PPOAgent(lr=0.0003, gamma=0.99,player_id=player_id, state_dim=BOARD_HEIGHT * BOARD_WIDTH, action_dim=BOARD_WIDTH
)
        elif name == "random":
            agent = RandomAgent(Current_Player=player_id)
        else:
            raise ValueError(f"Unknown agent type: {name}")
        agents.append(agent)
    return agents

def train(agent_names_args): # Changed agent_names to agent_names_args to match usage later
    agents = init_agents(agent_names_args)
    num_agents = len(agents)

    win_stats = {i + 1: 0 for i in range(num_agents)}
    win_stats['draw'] = 0
    total_moves = []
    agent_moves = {i + 1: [] for i in range(num_agents)}
    agent_epsilons = {i + 1: [] for i in range(num_agents)}
    agent_rewards = {i + 1: [] for i in range(num_agents)}
    agent_wins = {i + 1: [] for i in range(num_agents)}

    # --- Variables for saving the single overall best model ---
    overall_best_avg_win_rates = {i + 1: -1.0 for i in range(num_agents)}
    overall_best_model_episode = {i + 1: 0 for i in range(num_agents)}
    overall_best_model_state_dicts = {i + 1: None for i in range(num_agents)} # To store model state_dict in memory

    CHECKPOINT_WINDOW = 500
    os.makedirs("models", exist_ok=True)


    for episode in range(1, NUM_EPISODES + 1):
        game = Game(BOARD_HEIGHT, BOARD_WIDTH, num_agents, WIN_LENGTH)
        done = False
        moves = 0
        episode_moves = {i + 1: 0 for i in range(num_agents)}
        episode_rewards = {i + 1: 0.0 for i in range(num_agents)}

        while not done:
            current_agent = agents[game.current_player - 1]
            player_id = game.current_player
            state = current_agent.get_state(game)
            action = current_agent.select_action(game)

            if action is None:
                done = True
                break

            row, col = game.make_move(action)
            won = game.winning_moves(row, col)
            done = won or not game.get_valid_columns()

            reward = calculate_reward(game, player_id, row, col, done, reward_type='shaped')
            next_state = current_agent.get_state(game)

            if isinstance(current_agent, DoubleDQNAgent): # Ensure this check is appropriate for all your agent types
                current_agent.remember(state, action, reward, next_state, done)

            episode_rewards[player_id] += reward

            if hasattr(current_agent, 'observe') and getattr(current_agent, 'last_state', None) is not None:
                current_agent.observe(reward, game, done)
            elif hasattr(current_agent, 'store_outcome'):
                current_agent.store_outcome(game, row, col, done, reward)

            episode_moves[player_id] += 1



            if done:

                if game.winner:
                    agent_wins[game.winner].append(1)
                    for pid_inner in range(1, num_agents + 1): # Renamed pid to pid_inner to avoid conflict
                        if pid_inner != game.winner:
                            agent_wins[pid_inner].append(0) # Log 0 for losers
                else: # Draw
                    for pid_inner in range(1, num_agents + 1):
                        agent_wins[pid_inner].append(0) # Log 0 for all in case of a draw

                win_stats[game.winner or 'draw'] += 1
                break

            game.current_player = (game.current_player % num_agents) + 1
            moves += 1

        for agent in agents:
            if hasattr(agent, 'train') and callable(getattr(agent, 'train')): # Check if train is callable
                agent.train()

        for pid, agent in enumerate(agents, start=1):
            if hasattr(agent, 'epsilon'):
                agent_epsilons[pid].append(agent.epsilon)
                if agent.epsilon > getattr(agent, 'epsilon_min', 0.01):
                    agent.epsilon *= agent.epsilon_decay

        total_moves.append(moves)
        for pid in range(1, num_agents + 1):
            agent_moves[pid].append(episode_moves[pid])
            # Calculate average reward per move for the episode
            avg_episode_reward = episode_rewards[pid] / max(1, episode_moves[pid])
            agent_rewards[pid].append(avg_episode_reward)


        if episode % 100 == 0 or episode == 1:
            print(f"\n--- Episode {episode} ---")
            for pid in range(1, num_agents + 1):
                # Display current win rate for logging purposes
                current_wins_display = sum(agent_wins[pid][-CHECKPOINT_WINDOW:]) if len(agent_wins[pid]) >= CHECKPOINT_WINDOW else sum(agent_wins[pid])
                num_recent_games_display = min(len(agent_wins[pid]), CHECKPOINT_WINDOW) if CHECKPOINT_WINDOW <= len(agent_wins[pid]) else len(agent_wins[pid])
                current_avg_win_rate_display = (current_wins_display / num_recent_games_display) * 100 if num_recent_games_display > 0 else 0.0
                
                print(f"Player {pid} ({agent_names_args[pid - 1].upper()}): Wins = {win_stats[pid]}, Avg Moves (last 100) = {np.mean(agent_moves[pid][-100:]):.2f}, Recent Win Rate (last {num_recent_games_display}) = {current_avg_win_rate_display:.2f}%")
            print(f"Draws: {win_stats['draw']}, Avg Total Moves (last 100): {np.mean(total_moves[-100:]):.2f}")

            # --- Check for new overall best model (in memory) ---
            if episode >= CHECKPOINT_WINDOW:
                for pid in range(1, num_agents + 1):
                    if len(agent_wins[pid]) >= CHECKPOINT_WINDOW:
                        current_avg_win_rate = np.mean(agent_wins[pid][-CHECKPOINT_WINDOW:])
                        
                        # If this is the new best observed win rate, store its state in memory
                        if current_avg_win_rate > overall_best_avg_win_rates[pid]:
                            overall_best_avg_win_rates[pid] = current_avg_win_rate
                            overall_best_model_episode[pid] = episode
                            if hasattr(agents[pid-1], 'model') and hasattr(agents[pid-1].model, 'state_dict'):
                                overall_best_model_state_dicts[pid] = copy.deepcopy(agents[pid-1].model.state_dict())
                                print(f"Player {pid} ({agent_names_args[pid-1].upper()}) new peak performance: avg win rate {current_avg_win_rate*100:.2f}% (at ep {episode}). State captured.")
    
    print("\n======= Training Complete (End of Episodes) =======")
    for pid in range(1, num_agents + 1):
        print(f"Player {pid} ({agent_names_args[pid - 1].upper()}): Total Wins = {win_stats[pid]}, Avg Moves/Game = {np.mean(agent_moves[pid]):.2f}")
    print(f"Draws: {win_stats['draw']}")

    # --- Save final models with a timestamp (current state at end of training) ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n======= Saving Final Models with Timestamp: {timestamp} =======")
    for pid, agent in enumerate(agents, start=1):
        if hasattr(agent, 'save_model'):
            final_model_filename = f"models/{agent_names_args[pid - 1]}_agent_{pid}_final_{timestamp}.pkl"
            agent.save_model(final_model_filename)
            print(f"Saved final model: {final_model_filename}")
    
    # --- Save the single overall best models observed during training ---
    print("\n======= Saving Overall Best Models Observed During Training =======")
    for pid in range(1, num_agents + 1):
        if overall_best_model_state_dicts[pid] is not None:
            # Initialize a fresh agent instance for saving the best model state
            agent_key = agent_names_args[pid-1].lower()
            temp_agent_for_saving = None

            # Re-create agent with necessary parameters, similar to init_agents
            # Use benign values for learning-specific parameters if they are not needed for model structure
            if agent_key == "qlearn":
                temp_agent_for_saving = QlearnAgent(
                    learn_rate=0.0,  # Dummy value for saving
                    disc_factor=0.95, # Or a default/config value
                    explor_rate=0.0, # Dummy value
                    explor_decay=1.0, # Dummy value
                    player_id=pid
                )
            elif agent_key == "dqn":
                temp_agent_for_saving = DoubleDQNAgent(
                    board_height=BOARD_HEIGHT,
                    board_width=BOARD_WIDTH,
                    action_size=BOARD_WIDTH,
                    learning_rate=0.0,  # Dummy value for saving
                    gamma=0.99,         # Should match training or be a sensible default
                    epsilon=0.0,        # Dummy value
                    epsilon_min=0.0,    # Dummy value
                    epsilon_decay=1.0,  # Dummy value
                    player_id=pid
                )
            elif agent_key == "ppo":
                temp_agent_for_saving = PPOAgent(
                    player_id=pid,
                    state_dim=BOARD_HEIGHT * BOARD_WIDTH,
                    action_dim=BOARD_WIDTH,
                    lr=0.0,        # Dummy value for saving
                    gamma=0.99     # Should match training or be a sensible default
                )
            # elif agent_key == "random":
            #     # RandomAgent typically doesn't have a model state to save
            #     pass 
            else:
                print(f"Warning: Unknown agent type '{agent_key}' encountered when trying to save best model.")
                continue

            if temp_agent_for_saving and hasattr(temp_agent_for_saving, 'model') and hasattr(temp_agent_for_saving.model, 'load_state_dict'):
                temp_agent_for_saving.model.load_state_dict(overall_best_model_state_dicts[pid])
                overall_best_filename = f"models/overall_best_{agent_names_args[pid-1]}_agent_{pid}.pkl"
                temp_agent_for_saving.save_model(overall_best_filename)
                print(f"Saved overall best model for Player {pid} ({agent_names_args[pid-1].upper()}): {overall_best_filename} "
                      f"(achieved ~{overall_best_avg_win_rates[pid]*100:.2f}% win rate around episode {overall_best_model_episode[pid]})")
            else:
                print(f"Warning: Could not save overall best model for Player {pid} ({agent_names_args[pid-1].upper()}). Agent or model missing required methods.")
        else:
            print(f"No model state was captured as 'overall best' for Player {pid} ({agent_names_args[pid-1].upper()}). This might indicate an issue or very short training.")

    log_data = {
        **{f"agent{i + 1}_wins_raw": agent_wins[i + 1] for i in range(num_agents)},
        **{f"agent{i + 1}_wins": np.cumsum(agent_wins[i + 1]).tolist() for i in range(num_agents)},
        **{f"agent{i + 1}_moves": agent_moves[i + 1] for i in range(num_agents)},
        **{f"agent{i + 1}_epsilons": agent_epsilons[i + 1] for i in range(num_agents)},
        **{f"agent{i + 1}_rewards": agent_rewards[i + 1] for i in range(num_agents)},
        **{f"agent{i + 1}_name": agent_names_args[i] for i in range(num_agents)}, # Use agent_names_args
        "total_moves": total_moves,
        "num_episodes": NUM_EPISODES # Added num_episodes to log_data
    }

    with open("training_log.json", "w") as f:
        json.dump(log_data, f)

    subprocess.run(["python", "plot.py", "--log_file", "training_log.json"])

    print("\n======= Overall Training Summary =======")
    for pid in range(1, num_agents + 1):
        total_wins = win_stats[pid]
        win_rate = (total_wins / NUM_EPISODES) * 100
        print(f"Player {pid} ({agent_names_args[pid - 1].upper()}): Total Wins = {total_wins}, Win Rate = {win_rate:.2f}%, Avg Moves/Game = {np.mean(agent_moves[pid]):.2f}")
    print(f"Draws: {win_stats['draw']}")

    window = CHECKPOINT_WINDOW # Use the same window size
    print(f"\n======= Win Rate Last {window} Episodes (at end of training) =======")
    for pid in range(1, num_agents + 1):
        if len(agent_wins[pid]) >= window: # Check if enough data for the window
            recent_wins_count = sum(agent_wins[pid][-window:])
            recent_win_rate = (recent_wins_count / window) * 100
            print(f"Player {pid} ({agent_names_args[pid - 1].upper()}): Win Rate Last {window} Episodes = {recent_win_rate:.2f}%")
        elif len(agent_wins[pid]) > 0: # Handle cases with fewer than 'window' episodes
            recent_wins_count = sum(agent_wins[pid])
            recent_win_rate = (recent_wins_count / len(agent_wins[pid])) * 100
            print(f"Player {pid} ({agent_names_args[pid - 1].upper()}): Win Rate Last {len(agent_wins[pid])} Episodes = {recent_win_rate:.2f}%")
        else:
            print(f"Player {pid} ({agent_names_args[pid - 1].upper()}): No games played in the last window to calculate win rate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs='+', required=True, help="List of agent types (e.g. qlearn dqn)")
    # Added --episodes argument
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help=f"Number of episodes to train for (default: {NUM_EPISODES})")
    args = parser.parse_args()
    NUM_EPISODES = args.episodes # Update global NUM_EPISODES if provided
    train(args.agents)


