from game import Game
from Agents.double_q_learning import QlearnAgent
from Agents.double_dqn_agent import DoubleDQNAgent
from print import log
from Agents.ppo_agent import PPOAgent
# Game setup
BOARD_HEIGHT = 6
BOARD_WIDTH = 7
NUM_PLAYERS = 2
WINNING_LENGTH = 4
EPISODES = 30000



game = Game(BOARD_HEIGHT, BOARD_WIDTH, NUM_PLAYERS, WINNING_LENGTH)

# Load Q-learning agent
q_agent = QlearnAgent(
    learn_rate=0.0,
    disc_factor=0.95,
    explor_rate=0.0,
    explor_decay=1.0,
    player_id=1
)




dqn_agent = DoubleDQNAgent(
    board_height=BOARD_HEIGHT,
    board_width=BOARD_WIDTH,
    action_size=BOARD_WIDTH,
    player_id=0,              # NOT 0 — must match the second player
    learning_rate=0.0,        # Don’t train
    gamma=0.95,               # Doesn’t matter for inference
    epsilon=0.0,              # Always exploit learned policy
    epsilon_min=0.0,          # Not decaying anyway
    epsilon_decay=1.0         # Won’t change epsilon
)
dqn_agent.load_model("models/dqn_agent_1.pkl")



# Load PPO agent 
"""
ppo_agent = PPOAgent(
    player_id=1,
    state_dim=BOARD_HEIGHT * BOARD_WIDTH,
    action_dim=BOARD_WIDTH,
    lr=0.0,  # Inference only
    gamma=0.95
)
#ppo_agent.load_model("models/ppo_agent_2.pkl")  # Adjust model name as needed
"""

agents = [dqn_agent, q_agent]


results = {1: 0, 2: 0, "draw": 0}
print("main_testing.py: Game started!\n")

for episode in range(EPISODES):
    game = Game(BOARD_HEIGHT, BOARD_WIDTH, 2, WINNING_LENGTH)
    done = False

    while not done:
        agent = agents[game.current_player - 1]
        move = agent.select_action(game)
        if move is None:
            results["draw"] += 1
            break

        result = game.make_move(move)
        if not result:
            continue  # Invalid move, skip

        row, col = result
        if game.winning_moves(row, col):
            results[game.current_player] += 1
            break

        game.current_player = (game.current_player % 2) + 1

print(f"After {EPISODES} episodes:")
print(f"Player 1 ({agents[0].__class__.__name__}) wins: {results[1]}")
print(f"Player 2 ({agents[1].__class__.__name__}) wins: {results[2]}")
print(f"Draws: {results['draw']}")