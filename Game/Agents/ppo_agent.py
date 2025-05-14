import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from reward_utils import calculate_reward
import numpy as np
import os

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class PPOAgent:
    def __init__(self, player_id, state_dim, action_dim, lr, gamma, clip_epsilon=0.2):
        self.player_id = player_id
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        # Set device for GPU acceleration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network on GPU
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.memory = []
        self.losses = []

        print(f"PPOAgent initialized on {self.device}.")

    def select_action(self, game):
        # Flatten the board
        state = torch.tensor(game.board.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(state)
        
        valid_moves = game.get_valid_columns()

        if not valid_moves:
            return False
        
        # Mask invalid moves
        mask = torch.zeros_like(logits)
        mask[0, valid_moves] = 1
        logits = logits.masked_fill(mask == 0, -1e10)
        
        probs = torch.softmax(logits, dim=1)
        dist = Categorical(probs)
        action = dist.sample()

        # Save experience without reward and done (updated later)
        self.memory.append((state, action, dist.log_prob(action), None, None))

        return action.item()

    def store_outcome(self, game, row, col, done, reward):
        # Get the last stored experience
        last_state, last_action, last_log_prob, _, _ = self.memory[-1]


        # Update the experience with the calculated reward
        self.memory[-1] = (last_state, last_action, last_log_prob, reward, done)

    def train(self):
        # Check if we have any valid experiences
        if not self.memory or any(None in experience for experience in self.memory):
            return  # Don't train if we have incomplete experiences

        # Unpack memory
        states, actions, log_probs_old, rewards, dones = zip(*self.memory)

        # Compute returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        log_probs_old = torch.cat(log_probs_old).to(self.device)
        # Get current policy's log probs
        logits = self.policy(states)
        probs = torch.softmax(logits, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)

        # PPO objective
        ratios = torch.exp(log_probs - log_probs_old)
        advantages = returns - returns.mean()
        surrogate1 = ratios * advantages
        surrogate2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

        loss = -torch.min(surrogate1, surrogate2).mean()

        self.losses.append(loss.item())

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear memory
        self.memory = []



    def get_state(self, game):
        # Normalize the board: 0 = empty, 1 = own piece, -1 = opponent
        board_state = game.board.flatten().astype(np.float32)
        opponent_id = 3 - self.player_id
        board_state[board_state == self.player_id] = 1.0
        board_state[board_state == opponent_id] = -1.0
        return board_state



    def save_model(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.policy.state_dict(), file_path)
        print(f"PPOAgent: Model saved to {file_path}")

    def load_model(self, file_path):
        self.policy.load_state_dict(torch.load(file_path))
        self.policy.eval()
        print(f"PPOAgent: Model loaded from {file_path}")
