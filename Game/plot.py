import matplotlib.pyplot as plt
import numpy as np
import json
import os

def smooth(values, window=50):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')

def rolling_win_rate(wins, window=100):
    return [np.mean(wins[max(0, i - window):i+1]) for i in range(len(wins))]

def plot_agent_fig(agent_id, agent_name, data):
    episodes = list(range(1, len(data[f'agent{agent_id}_wins']) + 1))

    fig, axs = plt.subplots(4, 1, figsize=(10, 14))
    fig.suptitle(f"Agent {agent_id} ({agent_name.upper()}) Training Progress")

    axs[0].plot(episodes, rolling_win_rate(data[f'agent{agent_id}_wins']), label="Rolling Win Rate (100 eps)", color='green')
    axs[0].set_ylabel("Win Rate")
    axs[0].legend()

    smoothed_moves = smooth(data[f'agent{agent_id}_moves'])
    axs[1].plot(episodes[:len(smoothed_moves)], smoothed_moves, label="Smoothed Moves/Game", color='blue')
    axs[1].set_ylabel("Avg Moves")
    axs[1].legend()

    if f'agent{agent_id}_epsilons' in data:
        smoothed_eps = smooth(data[f'agent{agent_id}_epsilons'])
        axs[2].plot(episodes[:len(smoothed_eps)], smoothed_eps, label="Epsilon (Smoothed)", color='black')
        axs[2].set_ylabel("Epsilon")
        axs[2].legend()
    else:
        axs[2].text(0.5, 0.5, 'No Epsilon Data', horizontalalignment='center', verticalalignment='center')
        axs[2].set_ylabel("Epsilon")

    if f'agent{agent_id}_rewards' in data:
        smoothed_rewards = smooth(data[f'agent{agent_id}_rewards'])
        axs[3].plot(episodes[:len(smoothed_rewards)], smoothed_rewards, label="Smoothed Reward", color='cyan')
        axs[3].set_ylabel("Avg Reward")
        axs[3].legend()
    else:
        axs[3].text(0.5, 0.5, 'No Reward Data', horizontalalignment='center', verticalalignment='center')
        axs[3].set_ylabel("Avg Reward")

    axs[3].set_xlabel("Episodes")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"agent{agent_id}_{agent_name.lower()}_training.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to: {filename}")

def plot_training_curves(log_file_path):
    with open(log_file_path, 'r') as f:
        data = json.load(f)

    agent_ids = [int(k.replace("agent", "").split("_")[0]) for k in data if k.startswith("agent") and k.endswith("_wins")]
    agent_ids = sorted(set(agent_ids))

    for agent_id in agent_ids:
        name_key = f"agent{agent_id}_name"
        agent_name = data.get(name_key, f"agent{agent_id}")
        plot_agent_fig(agent_id, agent_name, data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default='training_log.json')
    args = parser.parse_args()

    plot_training_curves(args.log_file)