from game import Game
from Agents.double_q_learning import QlearnAgent
from Agents.double_dqn_agent import DoubleDQNAgent

# Game setup
BOARD_HEIGHT = 6
BOARD_WIDTH = 7
NUM_PLAYERS = 2
WINNING_LENGTH = 4

game = Game(BOARD_HEIGHT, BOARD_WIDTH, NUM_PLAYERS, WINNING_LENGTH)

# Load Q-learning agent
q_agent = QlearnAgent(
    learn_rate=0.0,
    disc_factor=0.95,
    explor_rate=0.0,
    explor_decay=1.0,
    player_id=1
)
q_agent.load_model("models/qlearn_agent_2.pkl")

# Agent list: Q-learning vs Human
#agents = [q_agent, "human"]



dqn_agent = DoubleDQNAgent(
    board_height=BOARD_HEIGHT,
    board_width=BOARD_WIDTH,
    action_size=BOARD_WIDTH,
    player_id=2,              # NOT 0 — must match the second player
    learning_rate=0.0,        # Don’t train
    gamma=0.95,               # Doesn’t matter for inference
    epsilon=0.0,              # Always exploit learned policy
    epsilon_min=0.0,          # Not decaying anyway
    epsilon_decay=1.0         # Won’t change epsilon
)

# Agent list: Q-learning vs Human
dqn_agent.load_model("models/dqn_agent_1.pkl")


agents = [q_agent, dqn_agent]

print("main.py: Game started!\n")

done = False
while not done:
    print(f"Current player: {game.current_player}")

    if agents[game.current_player - 1] == "human":
        while True:
            try:
                move = int(input(f"Your move (0-{game.board_width - 1}): "))
                if move in game.get_valid_columns():
                    break
                else:
                    print("Invalid move. Column full or out of range.")
            except ValueError:
                print("Please enter a valid integer.")

    elif agents[game.current_player - 1] == q_agent:
        agent = agents[game.current_player - 1]
        move = agent.select_action(game)

    elif agents[game.current_player - 1] == dqn_agent:
        agent = agents[game.current_player - 1]
        move = agent.select_action(game)

    if move is None:
        print("main.py: Board is full. It's a draw.")
        break

    result = game.make_move(move)
    if not result:
        print("main.py: Invalid move attempted. Try again.")
        continue

    row, col = result
    print(f"Player {game.current_player} played in column {col}, row {row}")
    game.print_board()
    print()

    if game.winning_moves(row, col):
        winner_agent = agents[game.current_player - 1]
        agent_type = "Q-learning" if winner_agent == q_agent else "DQN"
        print(f"main.py: Player {game.current_player} ({agent_type} agent) wins!\n")
        break


    game.current_player = (game.current_player % game.number_of_players) + 1
