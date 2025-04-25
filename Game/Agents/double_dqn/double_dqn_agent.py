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
    def __init__(self, player_id, board_height, board_width, action_size,
                 learning_rate=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000,
                 batch_size=64, update_target_freq=10,
                 reward_type='sparse'): # Added reward_type parameter
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
        self.reward_type = reward_type # Store reward type

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

        print(f"DoubleDQNAgent initialized on {self.device} with {self.reward_type} rewards.")


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
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.train_step_counter += 1
        if self.train_step_counter % self.update_target_freq == 0:
            self.update_target_network()


    def calculate_reward(self, game, row, col, done):
        """Calculates the reward based on the game state and reward type."""
        # --- Terminal Rewards (Common to both types) ---
        if done:
            if game.winner == self.player_id:
                return 10.0  # Win
            elif game.winner is not None: # Opponent won
                return -10.0 # Loss
            else: # Draw
                return 0.0

        # --- Intermediate Rewards (Only for 'shaped' type) ---
        if self.reward_type == 'shaped':
            intermediate_reward = 0.0
            opponent_id = 3 - self.player_id # Assuming player IDs 1 and 2

            # 1. Reward for creating 3-in-a-row (potential win setup)
            # Pass opponent_id to the helper function
            if self._check_line_length(game, row, col, self.player_id, game.winning_length - 1, opponent_id):
                intermediate_reward += 0.5 # Positive reward for setting up a win

            # 2. Reward for blocking opponent's 3-in-a-row (heuristic)
            # Check if placing the piece at (row, col) blocked a potential win for the opponent at that spot
            # Pass opponent_id to the helper function
            if self._check_line_length(game, row, col, self.player_id, game.winning_length - 1, opponent_id, check_for_player=opponent_id):
                 intermediate_reward += 0.4 # Positive reward for blocking

            # 3. Reward for playing in the center column (adjust index if needed)
            center_col = self.board_width // 2
            if col == center_col:
                intermediate_reward += 0.1

            # 4. Penalty for allowing opponent an immediate win next turn
            if self._check_opponent_immediate_win_threat(game, opponent_id):
                intermediate_reward -= 1.0 # Negative reward (penalty)

            return intermediate_reward
        else:
            # For 'sparse' rewards, return 0 for non-terminal states
            return 0.0

    # --- Helper methods for shaped rewards ---
    # Add opponent_id to the function signature
    def _check_line_length(self, game, row, col, player_id, length_needed, opponent_id, check_for_player=None):
        """
        Checks if the move at (row, col) by 'player_id' completed a line
        of exactly 'length_needed' for 'check_for_player'.
        If 'check_for_player' is None, it checks for 'player_id'.
        Used for detecting own 3-in-a-row or blocking opponent's 3-in-a-row.
        Requires opponent_id to check blocking condition.
        """
        if check_for_player is None:
            check_for_player = player_id

        # Temporarily place the piece to check lines passing through it
        # Note: The piece is already placed by game.make_move before calculate_reward is called.
        # So, we check lines passing through (row, col) for 'check_for_player'.

        board = game.board
        height = self.board_height
        width = self.board_width

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] # Horizontal, Vertical, Diag down-right, Diag up-right

        for dr, dc in directions:
            count = 1 # Count the piece at (row, col) itself if it matches check_for_player
            if board[row, col] != check_for_player:
                 # If the piece placed isn't the one we're checking for (e.g., checking opponent block)
                 # we need to count lines *towards* this spot from the opponent's perspective
                 count = 0 # Reset count, we only count neighbours

            # Count in positive direction (dr, dc)
            for i in range(1, length_needed):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < height and 0 <= c < width and board[r, c] == check_for_player:
                    count += 1
                else:
                    break
            # Count in negative direction (-dr, -dc)
            for i in range(1, length_needed):
                r, c = row - i * dr, col - i * dc
                if 0 <= r < height and 0 <= c < width and board[r, c] == check_for_player:
                    count += 1
                else:
                    break

            # Check if we found exactly the length needed
            # Important: For blocking check, we want to know if the opponent HAD length_needed-1 pointing here
            # Now opponent_id is defined within this function's scope
            if board[row, col] == player_id and check_for_player == opponent_id: # Agent just played, checking if opponent was blocked
                 # Opponent needed length_needed pieces aligned towards (row, col)
                 if count == length_needed -1:
                      return True # Agent blocked a line of required length-1
            elif board[row, col] == player_id and check_for_player == player_id: # Agent just played, checking own line
                 if count >= length_needed: # Check if agent created line of AT LEAST length_needed
                      # We need to be careful not to reward completing the winning line itself here,
                      # as that's covered by the terminal reward. Check for *exactly* length_needed?
                      # Let's check >= for now, simpler. Might need refinement.
                      # Check if this move was the winning move
                      if length_needed == game.winning_length: # Avoid double-counting win reward
                           continue
                      # Check if line is exactly length_needed? Or >= ?
                      # Let's check for exactly length_needed to be specific for 3-in-a-row reward
                      if count == length_needed:
                           # Further check: ensure it's an "open" line (can be extended) might be better
                           # For simplicity now, just check count.
                           return True
            # Add other cases if needed

        return False # No line of the specified length found passing through (row, col)

    def _check_opponent_immediate_win_threat(self, game, opponent_id):
        """
        Checks if the opponent has a winning move available on their next turn.
        This is checked *after* the agent has made its move.
        """
        valid_columns = game.get_valid_columns()
        original_board = game.board.copy() # Make a copy to simulate on

        for next_col in valid_columns:
            # Find the row the opponent's piece would land in
            next_row = -1
            for r in range(self.board_height - 1, -1, -1):
                if original_board[r, next_col] == 0:
                    next_row = r
                    break

            if next_row != -1:
                # Simulate placing the opponent's piece
                original_board[next_row, next_col] = opponent_id

                # Check if this simulated move is a win for the opponent
                # We need a way to check win condition without modifying the game state permanently
                # Let's borrow logic from game.winning_moves or create a static check method
                if self._static_check_win(original_board, opponent_id, next_row, next_col, game.winning_length):
                    # Found a winning move for the opponent
                    return True # Opponent has an immediate threat

                # Undo the simulation for the next check
                original_board[next_row, next_col] = 0

        return False # No immediate winning threat found for the opponent

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

    # ... save_model, load_model ...