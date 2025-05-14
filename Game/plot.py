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


def get_unique_plot_filename(base, ext=".png"):
    counter = 1
    path = f"{base}{ext}"
    while os.path.exists(path):
        path = f"{base}_{counter}{ext}"
        counter += 1
    return path

def plot_agent_fig(agent_id, agent_name, data):
    def win_rate_over_episodes(wins, window=100):
        win_diffs = np.diff([0] + wins)
        rates = [np.mean(win_diffs[max(0, i - window + 1):i + 1]) * 100 for i in range(len(win_diffs))]
        return rates
    def cumulative_win_rate(wins, window=100):
        rates = []
        for i in range(len(wins)):
            start = max(0, i - window + 1)
            prev = wins[start - 1] if start > 0 else 0
            count = wins[i] - prev
            win_rate = (count / (i - start + 1)) * 100
            rates.append(win_rate)
        return rates
    episodes = list(range(1, len(data[f'agent{agent_id}_wins']) + 1))

    fig, axs = plt.subplots(4, 1, figsize=(10, 16))
    fig.suptitle(f"Agent {agent_id} ({agent_name.upper()}) Training Progress")

    raw_wins = data[f'agent{agent_id}_wins_raw']
    win_rate = win_rate_over_episodes(raw_wins, window=100)
    win_rate = win_rate_over_episodes(data[f'agent{agent_id}_wins'], window=100)
    cumulative_win = cumulative_win_rate(data[f'agent{agent_id}_wins'])
    smoothed_win_rate = smooth(win_rate, window=50)
    axs[0].plot(episodes[:len(win_rate)], win_rate, label="Win Rate (%)", color='green')
    
    
    
    

    smoothed_moves = smooth(data[f'agent{agent_id}_moves'])
    axs[1].plot(episodes[:len(smoothed_moves)], smoothed_moves, label="Avg Moves/Game", color='blue')
    axs[1].set_ylabel("Avg Moves")
    axs[1].legend()
    
    
    axs[0].set_ylabel("Win Rate (%)")
    axs[0].set_ylim(0, 100)
    axs[0].legend()

    

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

    axs[3].set_xlabel("Training Episodes")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    base_name = f"agent{agent_id}_{agent_name.lower()}_training"
    filename = get_unique_plot_filename(base_name)
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
        agent_name = data.get(name_key, f"{data.get(f'agent{agent_id}_type', f'agent{agent_id}')}" )
        plot_agent_fig(agent_id, agent_name, data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default='training_log.json')
    args = parser.parse_args()

    plot_training_curves(args.log_file)
