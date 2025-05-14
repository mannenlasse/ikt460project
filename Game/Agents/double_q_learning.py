from .base_agent import Agent
import numpy as np
import random 
from collections import defaultdict
import pickle
import os

from print import log

class QlearnAgent(Agent):
    def __init__(self, learn_rate, disc_factor, explor_rate, explor_decay, epsilon_min=0.01, player_id = None):
        self.player_id = player_id 


        self.alpha = learn_rate  # learning rate
        self.gamma = disc_factor # discounting factor, how much do future rewards matter
        self.epsilon = explor_rate
        self.epsilon_decay = explor_decay
        self.epsilon_min = epsilon_min

        #parameters that will track last state and action, currently set to none and will be updated as the game moves on 
        self.last_state = None 
        self.last_action = None


        self.q1 = defaultdict(float)
        self.q2 = defaultdict(float)

    
    def get_state(self, game):
        return game.board

    def max_action(self, q, state, game):
        state_key = tuple(map(tuple, state))  # Convert to hashable format
        valid_actions = game.get_valid_columns()
        q_values = {a: q.get((state_key, a), 0.0) for a in valid_actions}

        max_q = max(q_values.values())
        best_actions = [a for a, val in q_values.items() if val == max_q]

        return random.choice(best_actions)



    # can also be said to be the decide_action because it decided what action the agent takes based on the current q-value estimation
    def select_action(self, game):

        #getting the game state in tuple
        state = self.get_state(game)
        #print(f"qlearning.py: select_Action_From_policy: {state}")

        #defining the set of actions (which column to drop)
        valid_actions = game.get_valid_columns()

        if not valid_actions:
            return None
        

        #greedy poilcy
        #exploration 
        if random.random() < self.epsilon:
            action = random.choice(valid_actions)
            log("qlearning.py: select_action [explore]: Random action selected due to exploration")
            return action

        #explotation
        else: 
            # Sum Q-values only for valid actions

            state_key = tuple(map(tuple, state))  # make it hashable
            q_sum = {a: self.q1.get((state_key, a), -0.1) + self.q2.get((state_key, a), -0.1) for a in valid_actions}

            if not q_sum:  # If no valid Q-values
                action = random.choice(valid_actions)
            else:
                max_q = max(q_sum.values())
                best_actions = [a for a, val in q_sum.items() if val == max_q]
                random.shuffle(best_actions)
                action = best_actions[0]
                                
            log("qlearning.py: select_action [exploit]: Best action selected")
            self.last_state = state
            self.last_action = action
            return action



    def observe(self, reward, game, done):

        next_state  = self.get_state(game)        
        valid_actions = game.get_valid_columns()
       
        if not valid_actions:
            log("heas no valid action")
            return
        

        rando = random.random()
        next_key = tuple(map(tuple, next_state))
        last_key = tuple(map(tuple, self.last_state))


        if rando < 0.5: 
            ############################################################################
            #    Q1(s, a) ← Q1(s, a) + α [r + γ * Q2(s', argmax_a Q1(s', a)) - Q1(s, a)]   
            ############################################################################

            #argmax_a Q1(s', a)
            a_ = self.max_action(self.q1, next_state, game)

            #Q2(s', argmax_a Q2(s', a))
            #future_q = self.q2.get((next_state, a_), 0.0)

            future_q = self.q2.get((next_key, a_), 0.0)
            #r + γ * Q2(s', argmax_a Q1(s', a))
            rewards_disc_future_q = reward + (0 if done else self.gamma * future_q) 

            #Q1(s, a)
            #old_q1 = self.q1.get((self.last_state, self.last_action), -0.0)
            old_q1 = self.q1.get((last_key, self.last_action), 0.0)

            #Q1(s, a) = Q1(s, a) + α(r + γ * Q2(s', argmax_a Q1(s', a)) - Q1(s, a))   
            self.q1[(last_key, self.last_action)] = old_q1 + self.alpha * (rewards_disc_future_q - old_q1)

        else:
            ############################################################################
            #    Q2(s, a) ← Q2(s, a) + α [r + γ * Q1(s', argmax_a Q2(s', a)) - Q2(s, a)]   
            ############################################################################

            #argmax_a Q2(s', a)
            a_ = self.max_action(self.q2, next_state, game)

            #Q2(s', argmax_a Q1(s', a))
            #future_q = self.q1.get((next_state, a_), 0.0)
            future_q = self.q1.get((next_key, a_), 0.0)
            #r + γ * Q2(s', argmax_a Q1(s', a))
            rewards_disc_future_q = reward + (0 if done else self.gamma * future_q)  
            
            #Q1(s, a)
            #old_q2 = self.q2.get((self.last_state, self.last_action), 0.0)
            old_q2 = self.q2.get((last_key, self.last_action), 0.0)

            #Q1(s, a) = Q1(s, a) + α(r + γ * Q2(s', argmax_a Q1(s', a)) - Q1(s, a))   
            self.q2[(last_key, self.last_action)]  = old_q2 + self.alpha *(rewards_disc_future_q - old_q2)





    def save_model(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({"q1": dict(self.q1), "q2": dict(self.q2)}, f)
        print(f"QlearnAgent: Q-tables saved to {file_path}")



    def load_model(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            self.q1 = defaultdict(float, data["q1"])
            self.q2 = defaultdict(float, data["q2"])
        print(f"QlearnAgent: Q-tables loaded from {file_path}")
