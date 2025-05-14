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

BOARD_HEIGHT = 6
BOARD_WIDTH = 7
WIN_LENGTH = 4
NUM_EPISODES = 5000

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
            agent = QlearnAgent(learn_rate=0.1, disc_factor=0.95, explor_rate=1.0, explor_decay=0.999, player_id=player_id)
        elif name == "dqn":
            agent = DoubleDQNAgent(board_height=BOARD_HEIGHT, board_width=BOARD_WIDTH, action_size=BOARD_WIDTH, player_id=player_id)
        elif name == "ppo":
            agent = PPOAgent(player_id=player_id, state_dim=BOARD_HEIGHT * BOARD_WIDTH, action_dim=BOARD_WIDTH)
        elif name == "random":
            agent = RandomAgent(Current_Player=player_id)
        else:
            raise ValueError(f"Unknown agent type: {name}")
        agents.append(agent)
    return agents

def train(agent_names):
    agents = init_agents(agent_names)
    num_agents = len(agents)

    win_stats = {i + 1: 0 for i in range(num_agents)}
    win_stats['draw'] = 0
    total_moves = []
    agent_moves = {i + 1: [] for i in range(num_agents)}
    agent_epsilons = {i + 1: [] for i in range(num_agents)}
    agent_rewards = {i + 1: [] for i in range(num_agents)}

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

            if isinstance(current_agent, DoubleDQNAgent):
                current_agent.remember(state, action, reward, next_state, done)

            episode_rewards[player_id] += reward

            if hasattr(current_agent, 'observe') and getattr(current_agent, 'last_state', None) is not None:
                current_agent.observe(reward, game, done)
            elif hasattr(current_agent, 'store_outcome'):
                current_agent.store_outcome(reward, done)

            episode_moves[player_id] += 1
            if done:
                win_stats[game.winner or 'draw'] += 1
                break

            game.current_player = (game.current_player % num_agents) + 1
            moves += 1

        for agent in agents:
            if hasattr(agent, 'train'):
                agent.train()

        for pid, agent in enumerate(agents, start=1):
            if hasattr(agent, 'epsilon'):
                agent_epsilons[pid].append(agent.epsilon)
                if agent.epsilon > getattr(agent, 'epsilon_min', 0.01):
                    agent.epsilon *= agent.epsilon_decay

        total_moves.append(moves)
        for pid in range(1, num_agents + 1):
            agent_moves[pid].append(episode_moves[pid])
            agent_rewards[pid].append(episode_rewards[pid] / max(1, episode_moves[pid]))

        if episode % 100 == 0 or episode == 1:
            print(f"\n--- Episode {episode} ---")
            for pid in range(1, num_agents + 1):
                print(f"Player {pid} ({agent_names[pid - 1].upper()}): Wins = {win_stats[pid]}, Avg Moves = {np.mean(agent_moves[pid][-100:]):.2f}")
            print(f"Draws: {win_stats['draw']}, Avg Total Moves (last 100): {np.mean(total_moves[-100:]):.2f}")

    print("\n======= Training Complete =======")
    for pid in range(1, num_agents + 1):
        print(f"Player {pid} ({agent_names[pid - 1].upper()}): Total Wins = {win_stats[pid]}, Avg Moves/Game = {np.mean(agent_moves[pid]):.2f}")
    print(f"Draws: {win_stats['draw']}")

    os.makedirs("models", exist_ok=True)
    for pid, agent in enumerate(agents, start=1):
        if hasattr(agent, 'save_model'):
            agent.save_model(f"models/{agent_names[pid - 1]}_agent_{pid}.pkl")

    log_data = {
        **{f"agent{i + 1}_wins": np.cumsum([1 if x < win_stats[i + 1] else 0 for x in range(NUM_EPISODES)]).tolist() for i in range(num_agents)},
        **{f"agent{i + 1}_moves": agent_moves[i + 1] for i in range(num_agents)},
        **{f"agent{i + 1}_epsilons": agent_epsilons[i + 1] for i in range(num_agents)},
        **{f"agent{i + 1}_rewards": agent_rewards[i + 1] for i in range(num_agents)},
        **{f"agent{i + 1}_name": agent_names[i] for i in range(num_agents)},
        "total_moves": total_moves
    }


    with open("training_log.json", "w") as f:
        json.dump(log_data, f)

    subprocess.run(["python", "plot.py", "--log_file", "training_log.json"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs='+', required=True, help="List of agent types (e.g. qlearn qlearn dqn dqn dqn)")
    args = parser.parse_args()
    train(args.agents)