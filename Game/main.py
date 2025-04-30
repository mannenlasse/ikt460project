import random
import matplotlib.pyplot as plt
from copy import deepcopy
from game import Game
from Agents.ppo_agent import PPOAgent
from Agents.random_agent import RandomAgent
from main_utils import count_winning_moves, opponent_has_winning_move

# Setup
state_dim = 6 * 7  # Flattened 6x7 board
action_dim = 7     # 7 possible columns
num_episodes = 5000

# Agents
agents = [PPOAgent(1, state_dim, action_dim), RandomAgent(2)]  # PPO Agent vs Random

# Tracking
all_rewards = []

ppo_wins = 0
ppo_losses = 0
draws = 0

print("Training started...\n")

for ep in range(num_episodes):
    game = Game(6, 7, 2, 4)
    done = False
    episode_reward = 0

    while not done:
        current_agent = agents[game.current_player - 1]
        opponent_agent = agents[game.current_player % 2]

        # Save previous state
        prev_game_state = deepcopy(game)

        # Agent selects action
        move = current_agent.select_action(game)
        if move is False:
            print(f"Episode {ep}: Board full. Draw.")
            break

        result = game.make_move(move)
        if not result:
            continue

        row, col = result
        won = game.winning_moves(row, col)
        draw = len(game.get_valid_columns()) == 0
        done = won or draw

        # --- Reward Shaping ---

        reward = 0
        opponent_id = (game.current_player % game.number_of_players) + 1

        agent_winning_moves = count_winning_moves(game, game.current_player)
        opponent_winning_moves = count_winning_moves(game, opponent_id)

        if won:
            reward = 1
        elif draw:
            reward = 0.5
        else:
            # +1 if agent forces certain win (>=2 winning moves)
            if agent_winning_moves >= 2:
                reward = 1

            # -1 if opponent could win after our move
            if opponent_winning_moves >= 1:
                reward -= 1

            # +0.5 if we blocked opponent's win
            if opponent_has_winning_move(prev_game_state, opponent_id) and not opponent_has_winning_move(game, opponent_id):
                reward += 0.5

            # -0.5 if we missed a winning move
            if agent_winning_moves >= 1:
                reward -= 0.5

        # --- Store outcome ---
        if isinstance(current_agent, PPOAgent):
            current_agent.store_outcome(reward, done)
        if isinstance(opponent_agent, PPOAgent):
            # Opponent also gets negative reward for our good actions
            opponent_reward = -reward
            opponent_agent.store_outcome(opponent_reward, done)

        episode_reward += reward

        # Switch player
        game.current_player = (game.current_player % game.number_of_players) + 1

    # After game ends: train PPO agent
    agents[0].train()
    all_rewards.append(episode_reward)

    if ep % 100 == 0:
        print(f"Episode {ep}: Reward {episode_reward:.2f}")

print("\nTraining completed!")

# --- Visualization ---
# Moving Average Rewards
window_size = 100
moving_avg_rewards = [
    sum(all_rewards[i-window_size:i]) / window_size
    if i >= window_size else
    sum(all_rewards[:i]) / (i if i > 0 else 1)
    for i in range(len(all_rewards))
]

# Plot
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Loss
axs[0].plot(agents[0].losses)
axs[0].set_title('PPO Loss Over Training')
axs[0].set_xlabel('Training Steps')
axs[0].set_ylabel('Loss')

# Average Reward
axs[1].plot(moving_avg_rewards)
axs[1].set_title('Moving Average Reward (window 100)')
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Average Reward')

plt.tight_layout()
plt.show()