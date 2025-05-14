import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os # Added

# Neural network for the Double DQN (Keep DQNNetwork class as is)
class DQNNetwork(nn.Module):
    def __init__(self, input_shape, output_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DoubleDQNAgent:
    # Remove reward_type from __init__ arguments and self.reward_type
    def __init__(self, board_height, board_width, action_size,
                 learning_rate=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.999, memory_size=10000,
                 batch_size=64, update_target_freq=10,  player_id = None):
        self.player_id = player_id
        self.board_height = board_height
        self.board_width = board_width
        self.state_size = board_height * board_width
        self.action_size = action_size

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        # self.reward_type = reward_type # Removed

        # Experience replay memory
        self.memory = deque(maxlen=memory_size)


        # Set device for GPU acceleration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Main network and target network (Use DQNNetwork)
        self.model = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = DQNNetwork(self.state_size, self.action_size).to(self.device)

        self.update_target_network() # Initialize target network

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training counter
        self.train_step_counter = 0

        # print(f"DoubleDQNAgent initialized on {self.device} with {self.reward_type} rewards.") # Removed reward type from message
        print(f"DoubleDQNAgent initialized on {self.device}.")


    def update_target_network(self):
        """Copies weights from the main network to the target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in replay memory."""
        # Ensure action is an integer index if it's not already
        action_index = int(action)
        self.memory.append((state, action_index, reward, next_state, done))


    def get_state(self, game):
        """Converts the game board into a flat numpy array state representation."""
        # Flatten the board and normalize (optional, but can help)
        # Normalization: 0 for empty, 1 for agent's piece, -1 for opponent's piece
        board_state = game.board.flatten().astype(np.float32)
        opponent_id = 3 - self.player_id # Assuming player IDs 1 and 2
        board_state[board_state == self.player_id] = 1.0
        board_state[board_state == opponent_id] = -1.0
        # Any remaining 0s are empty spots
        return board_state


    def select_action(self, game):
        """Selects an action using epsilon-greedy policy."""
        valid_columns = game.get_valid_columns()
        if not valid_columns:
            return None

        state = self.get_state(game)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if np.random.rand() <= self.epsilon:
            # Explore: choose a random valid action
            return random.choice(valid_columns)
        else:
            # Exploit: choose the best valid action according to the model
            self.model.eval() # Set model to evaluation mode
            with torch.no_grad():
                q_values = self.model(state_tensor).cpu().numpy()[0] # Get Q-values
            self.model.train() # Set model back to training mode

            # Filter Q-values for valid actions only
            valid_q_values = {col: q_values[col] for col in valid_columns}

            # Return the action (column) with the highest Q-value among valid actions
            if not valid_q_values: # Should not happen if valid_columns is not empty
                 return random.choice(valid_columns)
            return max(valid_q_values, key=valid_q_values.get)


    def train(self):
        """Trains the agent by replaying experiences from memory."""
        if len(self.memory) < self.batch_size:
            return # Not enough memory yet

        minibatch = random.sample(self.memory, self.batch_size)

        # Convert batch to tensors
        states = torch.FloatTensor(np.array([e[0] for e in minibatch])).to(self.device)
        actions = torch.LongTensor([e[1] for e in minibatch]).unsqueeze(1).to(self.device) # Ensure actions are Long and have shape [batch_size, 1]
        rewards = torch.FloatTensor([e[2] for e in minibatch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in minibatch])).to(self.device)
        dones = torch.BoolTensor([e[4] for e in minibatch]).unsqueeze(1).to(self.device) # Use BoolTensor

        # --- Double DQN Calculation ---
        # 1. Get Q values for current states from the main network
        current_q_values = self.model(states).gather(1, actions)

        # 2. Get the best action for next states using the main network
        next_actions = self.model(next_states).argmax(dim=1, keepdim=True)

        # 3. Get Q values for next states from the target network, using the action chosen by the main network
        next_q_values_target = self.target_model(next_states).gather(1, next_actions)

        # 4. Calculate target Q values: reward + gamma * next_q (if not done)
        target_q_values = rewards + (self.gamma * next_q_values_target * (~dones)) # Use ~dones as a mask (0 if done, 1 if not done)

        # 5. Calculate loss (MSE)
        loss = F.mse_loss(current_q_values, target_q_values.detach()) # Detach target to prevent gradient flow

        # 6. Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        # --- End Double DQN Calculation ---


        # Decay epsilon
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.train_step_counter += 1
        if self.train_step_counter % self.update_target_freq == 0:
            self.update_target_network()


    def _static_check_win(self, board, player_id, row, col, win_length):
        """
        Static check if placing player_id at (row, col) on the given board
        results in a win. Does not modify the board.
        """
        height, width = board.shape
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] # H, V, D\, D/

        for dr, dc in directions:
            count = 1
            # Count in positive direction
            for i in range(1, win_length):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < height and 0 <= c < width and board[r, c] == player_id:
                    count += 1
                else:
                    break
            # Count in negative direction
            for i in range(1, win_length):
                r, c = row - i * dr, col - i * dc
                if 0 <= r < height and 0 <= c < width and board[r, c] == player_id:
                    count += 1
                else:
                    break

            if count >= win_length:
                return True
        return False

    # --- Add save_model method ---
    def save_model(self, file_path):
        """Saves the current state of the main model."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Save the model's state dictionary
            torch.save(self.model.state_dict(), file_path)
            # Optional: Add a print statement here if you want confirmation from the agent itself
            # print(f"Agent {self.player_id}: Model state dict saved to {file_path}")
        except Exception as e:
            print(f"Error saving model for agent {self.player_id} to {file_path}: {e}")

    # --- Add load_model method ---
    def load_model(self, file_path):
        """Loads the model state from a file."""
        try:
            if os.path.exists(file_path):
                # Load the state dictionary and apply it to the model
                self.model.load_state_dict(torch.load(file_path, map_location=self.device))
                # Important: Update the target network to match the loaded model
                self.update_target_network()
                # Set model to evaluation mode if you are loading for inference/play
                self.model.eval()
                print(f"Agent {self.player_id}: Model state dict loaded from {file_path}")
            else:
                print(f"Error loading model for agent {self.player_id}: File not found at {file_path}")
        except Exception as e:
            print(f"Error loading model for agent {self.player_id} from {file_path}: {e}")