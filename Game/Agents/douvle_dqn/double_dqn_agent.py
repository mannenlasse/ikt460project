import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Neural network for the Double DQN
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
    def __init__(self, player_id, board_height, board_width, action_size, 
                 learning_rate=0.001, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000,
                 batch_size=64, update_target_freq=10):
        self.player_id = player_id
        self.board_height = board_height
        self.board_width = board_width
        self.state_size = board_height * board_width
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Main network and target network
        self.model = DQNNetwork(self.state_size, self.action_size)
        self.target_model = DQNNetwork(self.state_size, self.action_size)
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training counter
        self.train_step_counter = 0
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def get_state(self, game):
        # Convert the game board to a flat array for the neural network
        # We'll use 1 for our pieces, -1 for opponent pieces, and 0 for empty
        state = np.zeros((self.board_height, self.board_width))
        
        for i in range(self.board_height):
            for j in range(self.board_width):
                if game.board[i, j] == self.player_id:
                    state[i, j] = 1
                elif game.board[i, j] != 0:
                    state[i, j] = -1
        
        return state.flatten()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def select_action(self, game):
        # Get valid moves
        valid_columns = game.get_valid_columns()
        if not valid_columns:
            return None
        
        # Get current state
        state = self.get_state(game)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            # Explore: choose a random valid action
            return random.choice(valid_columns)
        else:
            # Exploit: choose the best action according to the model
            with torch.no_grad():
                q_values = self.model(state_tensor).detach().numpy()[0]
                
                # Filter for only valid actions
                valid_q_values = {col: q_values[col] for col in valid_columns}
                return max(valid_q_values, key=valid_q_values.get)
    
    def train(self):
        # Check if we have enough samples in memory
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            # Current Q values
            current_q = self.model(state_tensor)
            
            # Target Q values
            target_q = current_q.clone().detach()
            
            if done:
                target_q[0][action] = reward
            else:
                # Double DQN: use main network to select action, target network to evaluate
                with torch.no_grad():
                    # Select action using the main network
                    best_action = torch.argmax(self.model(next_state_tensor))
                    # Evaluate using the target network
                    next_q = self.target_model(next_state_tensor)[0][best_action]
                    target_q[0][action] = reward + self.gamma * next_q
            
            # Compute loss and update weights
            loss = F.mse_loss(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.train_step_counter += 1
        if self.train_step_counter % self.update_target_freq == 0:
            self.update_target_network()
    
    def calculate_reward(self, game, row, col, done):
        # Basic reward structure
        if done and game.winner == self.player_id:
            return 1.0  # Win
        elif done and game.winner is not None:
            return -1.0  # Loss
        elif done:
            return 0.0  # Draw
        
        # You can add more sophisticated rewards here, such as:
        # - Rewards for creating potential winning positions
        # - Penalties for allowing opponent to create winning positions
        # - Rewards for controlling the center columns
        
        return 0.0  # Default reward for non-terminal states
    
    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']