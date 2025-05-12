# Connect Four RL Agents

This project is a Connect Four environment with support for training reinforcement learning agents. The codebase includes implementations for Double DQN, Double Q-Learning, and PPO agents, along with a flexible training pipeline and utilities for reward shaping, evaluation, and plotting.

## Getting Started

To train an agent, you use the train.py script. This script is designed to be flexible, letting you choose which agent to train, what kind of reward structure to use, and what kind of opponent the agent should play against during training.

Here’s an example command to train a Double DQN agent with shaped rewards for 10,000 episodes, saving the model every 5,000 episodes, and using a pre-trained DQN model as the opponent:

python train.py --model dqn --reward_type shaped --episodes 10000 --save_freq 5000 --opponent dqn_model --opponent_model_path "/app/models/dqn_sparse_vs_random_ep50_20250429_012604.pt"

You can swap out the --model argument to train a different agent. The available options are:

dqn for Double DQN  
qlearn for Double Q-Learning  
ppo for PPO (this one is still being worked on)  

The --reward_type flag lets you choose between sparse and shaped rewards. Shaped rewards give the agent more feedback for intermediate moves, which can help it learn faster, especially on larger boards.

For the opponent, you can use a random agent or a pre-trained DQN model. If you want to train against a DQN opponent, just provide the path to the model checkpoint with --opponent_model_path.

All models and plots are saved in the models/ and plots/ directories. The training script will print progress and save plots of win rates, rewards, and other statistics at the end of training.

## Agents

Double DQN: Uses a neural network to estimate Q-values and supports experience replay and target networks. The implementation is in Game/Agents/double_dqn/double_dqn_agent.py.  
Double Q-Learning: A tabular agent that maintains two Q-tables and updates them using the Double Q-Learning algorithm. See Game/Agents/double_q_learning.py.  
PPO: Proximal Policy Optimization agent (work in progress).  

## Reward Structure

Rewards are calculated using the centralized function in Game/reward_utils.py. You can choose between sparse rewards (only for wins/losses/draws) or shaped rewards (which also give feedback for good moves, blocking, center column, and so on).

## Testing and Main Scripts

There are scripts for testing trained agents and for running games interactively (test.py and main.py), but these are still under development. More details and usage instructions will be added here once those scripts are finalized.

## Requirements

All dependencies are listed in requirements.txt. You can install them with:

pip install -r requirements.txt

