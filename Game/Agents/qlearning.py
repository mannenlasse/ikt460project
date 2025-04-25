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

        self.q_table1 = {}
        self.q_table2 = {}





    def get_state(self, game):
        return tuple(game.board.flatten())
    

    # can also be said to be the decide_action because it decided what action the agent takes based on the current q-value estimation
    def select_action_from_policy(self, game):
        state = self.get_state(game)

        #defining the set of actions (which column to drop)
        valid_actions = game.get_valid_columns()


        if random.random() < self.exploration_rate:
            action = random.choice(valid_actions)
            print(f"qlearning.py: select_action: [Explore] chose action {action} at random")

        else: 
            q_values = []

            for action in valid_actions: 
                q1_value = self.q_table1.get((state, action), 0.0)
                q2_value = self.q_table1.get((state, action), 0.0)

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


        # Estimate the best future Q-value from the next state
        if done:
            future_q = 0  # No future — game is over
        else:
            future_q = max(
                [self.q_table.get((next_state, a), 0.0) for a in next_game_state.get_valid_columns()],
                default=0.0
            )



        # Q-learning update rule
        updated_q = current_q + self.learning_rate * (reward + self.discouting_factor * future_q - current_q)
         

        self.q_table[(self.last_state, self.last_action)] = updated_q

        print(f"qlearning.py: observe: Updated Q[{self.last_state}, {self.last_action}] to {updated_q:.3f}")



