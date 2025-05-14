from game import Game
from Agents.double_q_learning import QlearnAgent

# Game setup
BOARD_HEIGHT = 6
BOARD_WIDTH = 7
NUM_PLAYERS = 2
WINNING_LENGTH = 4

game = Game(BOARD_HEIGHT, BOARD_WIDTH, NUM_PLAYERS, WINNING_LENGTH)

# Load Q-learning agent
q_agent = QlearnAgent(
    learn_rate=0.1,
    disc_factor=0.95,
    explor_rate=0.0,
    explor_decay=1.0,
    player_id=1
)
q_agent.load_model("models/qlearn_agent_2.pkl")

# Agent list: Q-learning vs Human
agents = [q_agent, "human"]

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
    else:
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
        print(f"main.py: Player {game.current_player} wins!\n")
        break

    game.current_player = (game.current_player % game.number_of_players) + 1
