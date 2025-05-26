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
EPISODES = 100



game = Game(BOARD_HEIGHT, BOARD_WIDTH, NUM_PLAYERS, WINNING_LENGTH)
"""
# Load Q-learning agent
q_agent = QlearnAgent(
    learn_rate=0.0,
    disc_factor=0.95,
    explor_rate=0.0,
    explor_decay=1.0,
    player_id=1
)
q_agent.load_model("models/dqn_agent_1.pkl")
"""



dqn_agent = DoubleDQNAgent(
    board_height=BOARD_HEIGHT,
    board_width=BOARD_WIDTH,
    action_size=BOARD_WIDTH,
    player_id=0,              
    learning_rate=0.0,        
    gamma=0.95,            
    epsilon=0.0,           
    epsilon_min=0.0,      
    epsilon_decay=1.0       
)
dqn_agent.load_model("models/dqn_agent_1.pkl")



# PPO agent
ppo_agent = PPOAgent(
    player_id=1,
    state_dim=BOARD_HEIGHT * BOARD_WIDTH,
    action_dim=BOARD_WIDTH,
    lr=0.0,  
    gamma=0.95
)
ppo_agent.load_model("models/ppo_agent_2.pkl")  


agents = [ppo_agent, dqn_agent]


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
            continue  

        row, col = result
        if game.winning_moves(row, col):
            results[game.current_player] += 1
            break

        game.current_player = (game.current_player % 2) + 1

print(f"After {EPISODES} episodes:")
print(f"Player 1 ({agents[0].__class__.__name__}) wins: {results[1]}")
print(f"Player 2 ({agents[1].__class__.__name__}) wins: {results[2]}")
print(f"Draws: {results['draw']}")