from .base_agent import Agent
import numpy as np
import random 


class QlearnAgent(Agent):
    def __init__(self, Current_Player, learn_rate = 0.1, disc_factor= 0.9, explor_rate = 1.0, explor_decay = 0.995):
    
        self.current_player = Current_Player

        self.learning_rate = learn_rate  # learning rate
        self.discouting_factor = disc_factor # discounting factor, how much do future rewards matter
        self.exploration_rate = explor_rate
        self.exploration_decay = explor_decay

        #parameters that will track last state and action, currently set to none and will be updated as the game moves on 
        self.last_state = None 
        self.last_action = None

        self.q_table = {}




    def get_state(self, game):
        return tuple(game.board.flatten())
    

    # can also be said to be the decide_action because it decided what action the agent takes based on the current q-value estimation
    def select_action(self, game):
        state = self.get_state(game)
        valid_actions = game.get_valid_columns()


        if random.random() < self.exploration_rate:
            action = random.choice(valid_actions)
            print(f"qlearning.py: select_action: [Explore] chose action {action} at random")

        else: 
            q_values = []

            for action in valid_actions: 
                q = self.q_table.get((state, action), 0.0)
                q_values.append((action, q))

            max_q = max(q_values, key=lambda x: x[1])[1]
            best_actions = [a for a, q in q_values if q == max_q]
            action = random.choice(best_actions)

            print(f"qlearning.py: select_action: [Exploit] chose best action {action} with Q={max_q:.3f}")


        self.last_state = state
        self.last_action = action

        return action



    def observe(self, reward, next_game_state, done):
        if self.last_state is None or self.last_action is None:
            return 
        
        next_state = tuple(next_game_state.board.flatten())

        current_q = self.q_table.get((self.last_state, self.last_action), 0.0)



         

    

