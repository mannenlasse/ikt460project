import os
import sys
import numpy as np
from collections import defaultdict

from Game.game import Game
from Game.Agents.random_agent import RandomAgent
from Game.Agents.double_dqn.double_dqn_agent import DoubleDQNAgent
from Game.Agents.double_q_learning import QlearnAgent
from Game.Agents.ppo_agent import PPOAgent


def load_agent(kind, model_path, player_id, board_height, board_width):
    if kind == "random":
        return RandomAgent(Current_Player=player_id)
    elif kind == "dqn":
        agent = DoubleDQNAgent(
            player_id=player_id,
            board_height=board_height,
            board_width=board_width,
            action_size=board_width,
            epsilon=0.0  # disable exploration
        )
        if model_path:
            agent.load_model(model_path)
        return agent
    elif kind == "qlearn":
        agent = QlearnAgent(
            player_id=player_id,
            learn_rate=0.0,
            disc_factor=0.99,
            explor_rate=0.0,
            explor_decay=1.0
        )
        if model_path:
            agent.load_model(model_path)
        return agent
    elif kind == "ppo":
        state_dim = board_height * board_width
        agent = PPOAgent(
            player_id=player_id,
            state_dim=state_dim,
            action_dim=board_width,
            lr=0.0
        )
        # (Optional) Load PPO model
        return agent
    else:
        raise ValueError(f"Unsupported agent type: {kind}")


def evaluate_multi_agent(opponent_defs, episodes, board_height, board_width, win_length):
    player_map = {}
    for i, (kind, model_path) in enumerate(opponent_defs):
        pid = i + 1
        player_map[pid] = load_agent(kind, model_path, pid, board_height, board_width)

    win_counts = defaultdict(int)
    draw_count = 0

    for ep in range(episodes):
        game = Game(board_height, board_width, len(player_map), win_length)
        done = False

        while not done:
            pid = game.current_player
            agent = player_map[pid]
            action = agent.select_action(game)

            if action is None:
                draw_count += 1
                break

            row, col = game.make_move(action)

            if game.winning_moves(row, col):
                win_counts[pid] += 1
                done = True
            elif not game.get_valid_columns():
                draw_count += 1
                done = True
            else:
                game.current_player = (game.current_player % len(player_map)) + 1

    print("\n--- Multi-Agent Evaluation Results ---")
    for pid in sorted(player_map):
        print(f"Player {pid} ({type(player_map[pid]).__name__}): {win_counts[pid]} wins")
    print(f"Draws: {draw_count}")
    print(f"Total Games: {episodes}")


if __name__ == "__main__":
    episodes = 5000
    board_height = 6
    board_width = 7
    win_length = 4

    agent_1 = ("random", None)

    # Define fixed opponents
    agent_2_or_more = [
        ("dqn", "models/dqn_shaped_vs_1xdqn_6x7_win4_ep5000.pt"),
        ("random", None)
    ]


    #("dqn", "models/dqn_shaped_vs_1xdqn_6x7_win4_ep5000.pt")
    #("qlearn", "models/qlearn_shaped_vs_1xdqn_6x7_win4_ep5000.pkl")
    opponent_defs = [agent_1] + agent_2_or_more

    evaluate_multi_agent(opponent_defs, episodes, board_height, board_width, win_length)