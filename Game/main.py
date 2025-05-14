import argparse
import os
from game import Game
from Agents.double_q_learning import QlearnAgent
from Agents.double_dqn_agent import DoubleDQNAgent
from Agents.ppo_agent import PPOAgent
from Agents.random_agent import RandomAgent

BOARD_HEIGHT = 6
BOARD_WIDTH = 7
NUM_PLAYERS = 2
WINNING_LENGTH = 4

def load_agent(spec: str, player_id: int):
    spec = spec.lower()

    if spec == "human":
        return "human", "HUMAN"

    if spec.endswith(".pkl") or spec.endswith(".pt"):
        name = os.path.basename(spec).lower()
        if "qlearn" in name:
            agent = QlearnAgent(learn_rate=0.0, disc_factor=0.95, explor_rate=0.0, explor_decay=1.0, player_id=player_id)
            agent.load_model(spec)
            return agent, "QLEARN"
        elif "dqn" in name:
            agent = DoubleDQNAgent(board_height=BOARD_HEIGHT, board_width=BOARD_WIDTH, action_size=BOARD_WIDTH,
                                   player_id=player_id, learning_rate=0.0, gamma=0.95,
                                   epsilon=0.0, epsilon_min=0.0, epsilon_decay=1.0)
            agent.load_model(spec)
            return agent, "DQN"
        elif "ppo" in name:
            agent = PPOAgent(player_id=player_id, state_dim=BOARD_HEIGHT * BOARD_WIDTH,
                             action_dim=BOARD_WIDTH, lr=0.0, gamma=0.95)
            agent.load_model(spec)
            return agent, "PPO"
        else:
            raise ValueError(f"Unknown agent model type from path: {spec}")
    else:
        if spec == "qlearn":
            return QlearnAgent(learn_rate=0.1, disc_factor=0.95, explor_rate=1.0, explor_decay=0.995, player_id=player_id), "QLEARN"
        elif spec == "dqn":
            return DoubleDQNAgent(board_height=BOARD_HEIGHT, board_width=BOARD_WIDTH, action_size=BOARD_WIDTH,
                                  player_id=player_id), "DQN"
        elif spec == "ppo":
            return PPOAgent(player_id=player_id, state_dim=BOARD_HEIGHT * BOARD_WIDTH,
                            action_dim=BOARD_WIDTH), "PPO"
        elif spec == "random":
            return RandomAgent(Current_Player=player_id), "RANDOM"
        else:
            raise ValueError(f"Unsupported agent type: {spec}")

def main(agent_specs):
    if len(agent_specs) != 2:
        raise ValueError("You must provide exactly two agents.")

    agents = []
    labels = []
    for i, spec in enumerate(agent_specs):
        agent, label = load_agent(spec, i + 1)
        agents.append(agent)
        labels.append(label)

    game = Game(BOARD_HEIGHT, BOARD_WIDTH, NUM_PLAYERS, WINNING_LENGTH)
    print("\nGame started!\n")

    done = False
    while not done:
        current_player = game.current_player
        agent = agents[current_player - 1]

        print(f"Current player: {current_player} ({labels[current_player - 1]})")

        if agent == "human":
            while True:
                try:
                    move = int(input(f"Your move (0-{game.board_width - 1}): "))
                    if move in game.get_valid_columns():
                        break
                    else:
                        print("Invalid move. Column full or out of range.")
                except ValueError:
                    print("Please enter a valid integer.")
        else:
            move = agent.select_action(game)

        if move is None:
            print("Board is full. It's a draw.")
            break

        result = game.make_move(move)
        if not result:
            print("Invalid move attempted. Try again.")
            continue

        row, col = result
        print(f"Player {current_player} played in column {col}, row {row}")
        game.print_board()
        print()

        if game.winning_moves(row, col):
            print(f"Player {current_player} ({labels[current_player - 1]}) wins!\n")
            break

        game.current_player = (game.current_player % game.number_of_players) + 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs=2, required=True, help="Specify two agents (e.g. qlearn dqn or paths to models)")
    args = parser.parse_args()
    main(args.agents)
