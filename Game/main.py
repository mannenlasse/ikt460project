import random
from game import Game
from Agents.random_agent import RandomAgent 
from Agents.agenta import Random_Agent_2



# height --- width --- numvber of players --- winning length #
game = Game(6, 7, 2, 4)


# which player gets wthich agent:  EXAMPLE: RandomAgent(0)  = Player 1 gets random agent  
agents = [RandomAgent(0), Random_Agent_2(1,0.3,0.4,0.5,0.6)]


print("main.py: Game started!\n")

done = False

while not done:

     
    
    print(f"Current player: {game.current_player}")


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


    if game.winning_moves(row, col):
        print(f"main.py: Player {game.current_player} has won the game!\n")
        done = True
        

    # Switch to next player
    game.current_player = (game.current_player % game.number_of_players) + 1
