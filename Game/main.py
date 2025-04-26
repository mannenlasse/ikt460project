import random
from game import Game
from Agents.random_agent import RandomAgent 
from Agents.agenta import Random_Agent_2
from Agents.qlearning import QlearnAgent



# height --- width --- numvber of players --- winning length #

# Track stats
wins = {1: 0, 2: 0}
draws = 0

NUM_EPISODES = 50
for episode in range(1, NUM_EPISODES + 1):
# which player gets wthich agent:  EXAMPLE: RandomAgent(0)  = Player 1 gets random agent  
#agents = [RandomAgent(0), Random_Agent_2(1,0.3,0.4,0.5,0.6)]

    game = Game(6, 7, 2, 5)

    agents = [RandomAgent(0), QlearnAgent(1, 0.1, 0.9, 0.8, 0.995)]


    print("main.py: Game started!\n")

    done = False
    turn_counter = 0

        
    while not done:

        turn_counter +=1

        print(f"\n\nmain.py: ==== Turn {turn_counter} ====")
        print(f"main.py: Current player: {game.current_player}")


        current_agent = agents[game.current_player - 1]

        move = current_agent.select_action(game)
        
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


        won = game.winning_moves(row, col)
        draw = len(game.get_valid_columns()) == 0
        done = won or draw

        reward = 1 if won else -0.05 if done else 0

        #parse rewards
        if won:
            print(f"Player {game.current_player} wins!")
            wins[game.current_player] += 1
        elif draw:
            print("It's a draw!")
            draws += 1

        current_agent.observe(reward, game, done)




        if won:
            print(f"main.py: Player {game.current_player} has won the game!\n")
            done = True
        elif draw:
            print("main.py: It's a DRAW.")

        # Switch to next player
        game.current_player = (game.current_player % game.number_of_players) + 1







# Final results
print("\n=== Training Summary ===")
print(f"RandomAgent (Player 1) wins: {wins[1]}")
print(f"QlearnAgent (Player 2) wins: {wins[2]}")
print(f"Draws: {draws}")