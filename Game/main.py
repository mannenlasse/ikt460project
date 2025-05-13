import random
from game import Game
from Agents.random_agent import RandomAgent
from Agents.agenta import Random_Agent_2
# Updated import path
from Agents.double_dqn.double_dqn_agent import DoubleDQNAgent
import os

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir = os.path.join(project_root, "models")

# Debug information
print(f"Project root: {project_root}")
print(f"Model directory: {model_dir}")
print(f"Model directory exists: {os.path.exists(model_dir)}")

board_height = 6
board_width = 7
number_of_players = 2
winnding_length = 4

# height --- width --- numvber of players --- winning length #
game = Game(board_height, board_width, number_of_players, winnding_length)

# Legg til en menneskelig spiller
def human_player(game):
    while True:
        try:
            move = int(input(f"Din tur! Velg kolonne (0-{game.board_width-1}): "))
            if 0 <= move < game.board_width:
                return move
            else:
                print(f"Ugyldig trekk. Velg en kolonne mellom 0 og {game.board_width-1}.")
        except ValueError:
            print("Vennligst skriv inn et tall.")

# Initialiser første AI-agent
dqn_agent1 = DoubleDQNAgent(player_id=1, action_size=game.board_width, 
                          board_height=game.board_height, board_width=game.board_width)
# Use absolute path for model loading
model1_path = os.path.join(model_dir, "dqn_shaped_vs_1xdqn_6x7_win4_ep5000.pt")
print(f"Model 1 path: {model1_path}")
print(f"Model 1 exists: {os.path.exists(model1_path)}")
dqn_agent1.load_model(model1_path)
dqn_agent1.epsilon = 0.0  # Ingen utforskning, kun utnyttelse

# Initialiser andre AI-agent
dqn_agent2 = DoubleDQNAgent(player_id=2, action_size=game.board_width, 
                          board_height=game.board_height, board_width=game.board_width)
# Use absolute path for model loading
model2_path = os.path.join(model_dir, "dqn_shaped_vs_1xdqn_6x7_win4_ep5000.pt")
print(f"Model 2 path: {model2_path}")
print(f"Model 2 exists: {os.path.exists(model2_path)}")
dqn_agent2.load_model(model2_path)
dqn_agent2.epsilon = 0.0  # Ingen utforskning, kun utnyttelse

# Velg spillmodus
print("Velg spillmodus:")
print("1: Spill mot AI (du er spiller 2)")
print("2: La to AI-er spille mot hverandre")
while True:
    try:
        mode = int(input("Skriv inn ditt valg (1 eller 2): "))
        if mode in [1, 2]:
            break
        else:
            print("Ugyldig valg. Skriv inn 1 eller 2.")
    except ValueError:
        print("Vennligst skriv inn et tall.")

# Sett opp agenter basert på valgt modus
if mode == 1:
    agents = [dqn_agent1, human_player]
    print("\nDu spiller som spiller 2. AI er spiller 1.")
else:
    agents = [dqn_agent1, dqn_agent2]
    print("\nAI 1 vs AI 2 modus valgt.")

print("main.py: Game started!\n")
print(f"Brettet er {board_height} rader høyt og {board_width} kolonner bredt.")
print(f"Du må få {winnding_length} på rad for å vinne.\n")

done = False

while not done:
    
    print(f"Current player: {game.current_player}")

    current_agent = agents[game.current_player - 1]
    
    if callable(current_agent):
        # For menneskelig spiller
        move = current_agent(game)
    else:
        # For AI-agent
        move = current_agent.select_action(game)
        if mode == 2:
            # Legg til en liten pause for å kunne følge med på AI vs AI spillet
            input("Trykk Enter for å se neste trekk...")
    
    if move is None:
        print("main.py: Board is full. It's a draw.")
        break

    result = game.make_move(move)

    if not result:
        continue  # Try again if move failed

    row, col = result
    
    print(f"main.py: Player {game.current_player} played in column {move} and col: {col} and row: {row} and result: {result}")
    game.print_board()
    print("")

    if game.winning_moves(row, col):
        print(f"main.py: Player {game.current_player} has won the game!\n")
        done = True
        
    # Switch to next player
    game.current_player = (game.current_player % game.number_of_players) + 1
