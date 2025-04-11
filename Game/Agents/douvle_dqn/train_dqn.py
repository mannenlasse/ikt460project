# Change these import lines
import numpy as np
import sys
import os
import torch  # Legg til denne importen

# Add the project root to the path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from Game.game import Game
from Game.Agents.random_agent import RandomAgent
from Game.Agents.douvle_dqn.double_dqn_agent import DoubleDQNAgent
import matplotlib.pyplot as plt
import os

# Create directory for saving models
os.makedirs('models', exist_ok=True)

# Game parameters
BOARD_HEIGHT = 6
BOARD_WIDTH = 7
NUM_PLAYERS = 2
WIN_LENGTH = 4
NUM_EPISODES = 10000
SAVE_FREQUENCY = 500

# Initialize agents
dqn_agent = DoubleDQNAgent(
    player_id=1,
    board_height=BOARD_HEIGHT,
    board_width=BOARD_WIDTH,
    action_size=BOARD_WIDTH,
    memory_size=50000,
    batch_size=64
)

# Legg til GPU-akselerasjon hvis tilgjengelig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dqn_agent.model.to(device)
dqn_agent.target_model.to(device)

# Oppdater også select_action og train metodene i DoubleDQNAgent-klassen for å bruke device
# Dette kan gjøres ved å modifisere double_dqn_agent.py

opponent = RandomAgent(1)  # Will be assigned player_id 2 in the game loop

# Statistics tracking
win_history = []
loss_history = []
draw_history = []
reward_history = []
epsilon_history = []

# Training loop
for episode in range(NUM_EPISODES):
    # Create a new game
    game = Game(BOARD_HEIGHT, BOARD_WIDTH, NUM_PLAYERS, WIN_LENGTH)
    
    # Reset episode variables
    done = False
    total_reward = 0
    
    # Store game history for training
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    
    # Game loop
    while not done:
        current_player_id = game.current_player
        
        # Get current state
        current_state = dqn_agent.get_state(game) if current_player_id == 1 else None
        
        # Select action based on current player
        if current_player_id == 1:  # DQN agent's turn
            action = dqn_agent.select_action(game)
        else:  # Opponent's turn
            opponent.player_id = current_player_id - 1  # Adjust player_id for the opponent
            action = opponent.select_action(game)
        
        # Check if the board is full
        if action is None:
            print(f"Episode {episode}: Game ended in a draw (board full)")
            draw_history.append(1)
            win_history.append(0)
            loss_history.append(0)
            done = True
            break
        
        # Make the move
        result = game.make_move(action)
        
        # If move failed, try again
        if not result:
            continue
        
        row, col = result
        
        # Check if the game is over
        if game.winning_moves(row, col):
            done = True
            if game.winner == 1:
                print(f"Episode {episode}: DQN agent won!")
                win_history.append(1)
                loss_history.append(0)
                draw_history.append(0)
            else:
                print(f"Episode {episode}: Opponent won!")
                win_history.append(0)
                loss_history.append(1)
                draw_history.append(0)
        
        # If the board is full after this move, it's a draw
        if not game.get_valid_columns() and not done:
            print(f"Episode {episode}: Game ended in a draw (no valid moves)")
            draw_history.append(1)
            win_history.append(0)
            loss_history.append(0)
            done = True
        
        # If it was the DQN agent's turn, store the transition
        if current_player_id == 1:
            next_state = dqn_agent.get_state(game)
            reward = dqn_agent.calculate_reward(game, row, col, done)
            total_reward += reward
            
            # Store transition
            states.append(current_state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Switch to next player if game is not done
        if not done:
            game.current_player = (game.current_player % game.number_of_players) + 1
    
    # After the episode, add all transitions to the replay buffer
    for i in range(len(states)):
        dqn_agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
    
    # Train the DQN agent
    dqn_agent.train()
    
    # Record statistics
    reward_history.append(total_reward)
    epsilon_history.append(dqn_agent.epsilon)
    
    # Print progress
    if (episode + 1) % 100 == 0:
        win_rate = sum(win_history[-100:]) / 100
        loss_rate = sum(loss_history[-100:]) / 100
        draw_rate = sum(draw_history[-100:]) / 100
        avg_reward = sum(reward_history[-100:]) / 100
        
        print(f"Episode {episode + 1}/{NUM_EPISODES}")
        print(f"Win Rate: {win_rate:.2f}, Loss Rate: {loss_rate:.2f}, Draw Rate: {draw_rate:.2f}")
        print(f"Average Reward: {avg_reward:.4f}, Epsilon: {dqn_agent.epsilon:.4f}")
        print("-" * 50)
    
    # Save model periodically
    if (episode + 1) % SAVE_FREQUENCY == 0:
        dqn_agent.save_model(f"models/dqn_agent_episode_{episode + 1}.pt")

# Save the final model
dqn_agent.save_model("models/dqn_agent_final.pt")

# Plot training statistics
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(range(len(win_history)), win_history, 'g-', alpha=0.3)
plt.plot(np.convolve(win_history, np.ones(100)/100, mode='valid'), 'g-')
plt.title('Win Rate')
plt.xlabel('Episode')
plt.ylabel('Win (1) / Loss (0)')

plt.subplot(2, 2, 2)
plt.plot(range(len(reward_history)), reward_history, 'b-', alpha=0.3)
plt.plot(np.convolve(reward_history, np.ones(100)/100, mode='valid'), 'b-')
plt.title('Reward History')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(2, 2, 3)
plt.plot(range(len(epsilon_history)), epsilon_history, 'r-')
plt.title('Epsilon Decay')
plt.xlabel('Episode')
plt.ylabel('Epsilon')

plt.subplot(2, 2, 4)
window_size = 100
win_rate_history = [sum(win_history[max(0, i-window_size):i])/min(i, window_size) for i in range(1, len(win_history)+1)]
loss_rate_history = [sum(loss_history[max(0, i-window_size):i])/min(i, window_size) for i in range(1, len(loss_history)+1)]
draw_rate_history = [sum(draw_history[max(0, i-window_size):i])/min(i, window_size) for i in range(1, len(draw_history)+1)]

plt.plot(range(len(win_rate_history)), win_rate_history, 'g-', label='Win Rate')
plt.plot(range(len(loss_rate_history)), loss_rate_history, 'r-', label='Loss Rate')
plt.plot(range(len(draw_rate_history)), draw_rate_history, 'b-', label='Draw Rate')
plt.title('Performance Metrics')
plt.xlabel('Episode')
plt.ylabel('Rate')
plt.legend()

plt.tight_layout()
plt.savefig('training_stats.png')
plt.show()