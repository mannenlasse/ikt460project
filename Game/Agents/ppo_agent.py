import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

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
    def __init__(self, player_id, state_dim, action_dim, lr=1e-3, gamma=0.99, clip_epsilon=0.2):
        self.player_id = player_id
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Memory for training
        self.memory = []
        
        # Saving loss for plotting
        self.losses = []

    def select_action(self, game):
        # Flatten the board
        state = torch.FloatTensor(game.board.flatten()).unsqueeze(0)  # (1, state_dim)
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

        self.memory.append((state, action, dist.log_prob(action)))
        
        return action.item()

    def store_outcome(self, reward, done):
        self.memory[-1] += (reward, done)

    def train(self):
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

        returns = torch.FloatTensor(returns)
        states = torch.cat(states)
        actions = torch.LongTensor(actions)
        log_probs_old = torch.cat(log_probs_old)

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