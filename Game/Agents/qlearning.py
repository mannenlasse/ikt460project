from .base_agent import Agent
import numpy as np
import random 


class QlearnAgent(Agent):
    def __init__(self, Current_Player, learn_rate, disc_factor, explor_rate, explor_decay):
    
        self.current_player = Current_Player

        self.learning_rate = learn_rate  # learning rate
        self.discouting_factor = disc_factor # discounting factor, how much do future rewards matter
        self.exploration_rate = explor_rate
        self.exploration_decay = explor_decay

        #parameters that will track last state and action, currently set to none and will be updated as the game moves on 
        self.last_state = None 
        self.last_action = None


        self.q1 = {}
        self.q2 = {} 

       


    def get_state(self, game):
        return tuple(game.board.flatten())


    def max_action(self, game):
        #getting the game state in tuple
        state = self.get_state(game)
        #print(f"qlearning.py: select_Action_From_policy: {state}")

        #defining the set of actions (which column to drop)
        valid_actions = game.get_valid_columns()
  
        q_values = [self.q1.get((state, action)) + self.q2.get((state, action)) for action in valid_actions]
        
        max_q = max(q_values)
        best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        action = random.choice(best_actions)

        return action


    # can also be said to be the decide_action because it decided what action the agent takes based on the current q-value estimation
    def select_action_from_policy(self, game):

        #getting the game state in tuple
        state = self.get_state(game)
        #print(f"qlearning.py: select_Action_From_policy: {state}")

        #defining the set of actions (which column to drop)
        valid_actions = game.get_valid_columns()


        #greedy poilcy
        #exploration 
        if random.random() < self.exploration_rate:
            action = random.choice(valid_actions)
            print("qlearning.py: select_action: Random action selected due to exploration")

        #explotation
        else: 
            print(f"heidu")

            #retrieving q values from table 1 and 2 and summing them up and storing in a list called q values
            q_values = [self.q1.get((state, action)) + self.q2.get((state, action)) for action in valid_actions]
           
            max_q = max(q_values)
            best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
            action = random.choice(best_actions)
            print("qlearning.py: select_action: Best action selected")

        self.last_state = state
        self.last_action = action
        return action
    



    def observe(self, reward, next_game_state, done):
        if self.last_state is None or self.last_action is None:
            return

        next_state = tuple(next_game_state.board.flatten())
        valid_actions = next_game_state.get_valid_columns()

        # Randomly choose whether to update q_table1 or q_table2
        if random.random() < 0.5:
            print("fs")
            # Update Q1 using Q2 for qlearning.py: select_Action_From_policy: 



    def select_action(self, game):
        return self.select_action_from_policy(game)
