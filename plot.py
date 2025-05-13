# plot.py

import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

def get_next_version(base_filename):
    i = 1
    while True:
        if i == 1:
            test_filename = base_filename
        else:
            name_parts = base_filename.rsplit('.', 1)
            test_filename = f"{name_parts[0]}_v{i}.{name_parts[1]}"
        if not os.path.exists(test_filename):
            return test_filename
        i += 1

def moving_average(data, window):
    if len(data) == 0:
        return np.array([])
    actual_window = min(len(data), window)
    return np.convolve(data, np.ones(actual_window)/actual_window, mode='valid')

def generate_training_plots(
    win_history,
    loss_history,
    draw_history,
    reward_history,
    epsilon_history,
    opponent_definitions,
    model_type,
    reward_type,
    board_height,
    board_width,
    win_length,
    num_episodes,
    plot_dir
):
    plt.figure(figsize=(15, 10))
    window_size = 100
    episodes_axis = np.arange(1, num_episodes + 1)

    ma_win_rate = moving_average(win_history, window_size)
    ma_reward = moving_average(reward_history, window_size)
    ma_episodes = episodes_axis[window_size-1:] if len(ma_win_rate) > 0 else np.array([])

    # --- Win Rate Plot ---
    plt.subplot(2, 2, 1)
    if len(ma_episodes) > 0:
        plt.plot(ma_episodes, ma_win_rate, label=f'Win Rate (MA {window_size})', color='green')
        plt.plot(episodes_axis, win_history, alpha=0.3, color='lightgreen')
    else:
        plt.plot(episodes_axis, win_history, label='Win Rate', color='green')
    plt.title('Win Rate')
    plt.xlabel('Episodes')
    plt.ylabel('Rate')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)

    # --- Reward Plot ---
    plt.subplot(2, 2, 2)
    if len(ma_episodes) > 0:
        plt.plot(ma_episodes, ma_reward, label=f'Avg Reward (MA {window_size})', color='purple')
        plt.plot(episodes_axis, reward_history, alpha=0.3, color='plum')
    else:
        plt.plot(episodes_axis, reward_history, label='Avg Reward', color='purple')
    plt.title('Average Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)

    # --- Epsilon Plot ---
    plt.subplot(2, 2, 3)
    if epsilon_history and len(episodes_axis) > 0:
        plt.plot(episodes_axis, epsilon_history, label='Epsilon', color='red', linestyle=':')
        plt.ylabel('Epsilon')
    else:
        plt.text(0.5, 0.5, 'No Epsilon data', ha='center', va='center')
    plt.title('Epsilon Decay')
    plt.xlabel('Episodes')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)

    # --- Learning Progress Plot ---
    plt.subplot(2, 2, 4)
    if len(win_history) > window_size:
        improvement = []
        for i in range(window_size, len(win_history), window_size):
            prev = np.mean(win_history[i-window_size:i])
            curr = np.mean(win_history[i:min(i+window_size, len(win_history))])
            improvement.append(curr - prev)

        x = np.arange(window_size, len(win_history), window_size)[:len(improvement)]
        plt.bar(x, improvement, width=window_size * 0.8, color='purple')
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        max_abs_change = max(abs(min(improvement)), abs(max(improvement))) if improvement else 0.2
        y_limit = min(0.3, round(max_abs_change * 1.2 * 20) / 20)
        plt.ylim(-y_limit, y_limit)
        plt.title('Learning Progress')
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate Change')
        plt.grid(True, axis='y')
    else:
        plt.text(0.5, 0.5, 'Not enough data for learning progress', ha='center', va='center')
        plt.title('Learning Progress')
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate Change')

    # --- Save Plot ---
    opponent_types = [kind for kind, _ in opponent_definitions]
    opponent_counts = Counter(opponent_types)
    opponent_desc = "_".join(f"{count}x{kind}" for kind, count in opponent_counts.items())
    board_info = f"{board_height}x{board_width}"
    win_info = f"win{win_length}"
    filename = f'plot_{model_type}_{reward_type}_vs_{opponent_desc}_{board_info}_{win_info}_ep{num_episodes}.png'

    os.makedirs(plot_dir, exist_ok=True)
    full_path = get_next_version(os.path.join(plot_dir, filename))
    plt.suptitle(f'Training Performance: {model_type.upper()} ({reward_type}) vs {opponent_desc} ({board_info}, {num_episodes} Episodes)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(full_path)
    print(f"Plot saved to: {full_path}")
