import numpy as np

# --- Helper Functions (Moved from DoubleDQNAgent, 'self' removed) ---

def _static_check_win(board, player_id, row, col, win_length):
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

def _check_line_length(game, row, col, player_id, length_needed, opponent_id, check_for_player=None):
    """
    Checks if the move at (row, col) by 'player_id' completed a line
    of exactly 'length_needed' for 'check_for_player'.
    If 'check_for_player' is None, it checks for 'player_id'.
    Used for detecting own 3-in-a-row or blocking opponent's 3-in-a-row.
    Requires opponent_id to check blocking condition.
    """
    if check_for_player is None:
        check_for_player = player_id

    board = game.board
    height = game.board_height # Use game attributes
    width = game.board_width  # Use game attributes

    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in directions:
        count = 1
        if board[row, col] != check_for_player:
             count = 0

        # Count in positive direction
        for i in range(1, length_needed + 1): # Check up to length_needed
            r, c = row + i * dr, col + i * dc
            if 0 <= r < height and 0 <= c < width and board[r, c] == check_for_player:
                count += 1
            else:
                break
        # Count in negative direction
        for i in range(1, length_needed + 1): # Check up to length_needed
            r, c = row - i * dr, col - i * dc
            if 0 <= r < height and 0 <= c < width and board[r, c] == check_for_player:
                count += 1
            else:
                break

        # Adjust count logic slightly as we counted both directions from center
        # Total line length including the center piece (if it matches check_for_player)
        # Recalculate count across the whole line through (row, col)
        line_count = 1 if board[row, col] == check_for_player else 0
        # Positive direction
        for i in range(1, game.winning_length):
             r, c = row + i * dr, col + i * dc
             if 0 <= r < height and 0 <= c < width and board[r, c] == check_for_player:
                 line_count += 1
             else:
                 break
        # Negative direction
        for i in range(1, game.winning_length):
             r, c = row - i * dr, col - i * dc
             if 0 <= r < height and 0 <= c < width and board[r, c] == check_for_player:
                 line_count += 1
             else:
                 break


        # Check blocking condition: Did the opponent have length_needed-1 before this move?
        if board[row, col] == player_id and check_for_player == opponent_id:
            # Check count in positive direction *before* the move
            count_pos = 0
            for i in range(1, length_needed):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < height and 0 <= c < width and board[r, c] == opponent_id:
                    count_pos += 1
                else: break
            # Check count in negative direction *before* the move
            count_neg = 0
            for i in range(1, length_needed):
                r, c = row - i * dr, col - i * dc
                if 0 <= r < height and 0 <= c < width and board[r, c] == opponent_id:
                    count_neg += 1
                else: break

            if count_pos + count_neg >= length_needed - 1:
                 # Check if the space at (row, col) was the only empty spot completing the line
                 # This logic might need refinement, but let's assume blocking if count was >= length_needed - 1
                 return True # Agent potentially blocked a line

        # Check own line creation condition
        elif board[row, col] == player_id and check_for_player == player_id:
             # Check if agent created line of AT LEAST length_needed
             # Avoid double-counting win reward
             if length_needed == game.winning_length:
                  continue
             # Check for exactly length_needed for the intermediate reward
             if line_count == length_needed:
                  return True

    return False



def _is_diagonal_threat(game, row, col, player_id, length_needed):
    board = game.board
    height = game.board_height
    width = game.board_width
    directions = [(1, 1), (1, -1)]  # Diagonals only

    for dr, dc in directions:
        count = 1
        for i in range(1, length_needed):
            r, c = row + i * dr, col + i * dc
            if 0 <= r < height and 0 <= c < width and board[r, c] == player_id:
                count += 1
            else:
                break
        for i in range(1, length_needed):
            r, c = row - i * dr, col - i * dc
            if 0 <= r < height and 0 <= c < width and board[r, c] == player_id:
                count += 1
            else:
                break
        if count >= length_needed:
            return True
    return False



def _check_opponent_immediate_win_threat(game, opponent_id):
    """
    Checks if the opponent has a winning move available on their next turn.
    """
    valid_columns = game.get_valid_columns()
    original_board = game.board.copy()
    board_height = game.board_height # Use game attributes

    for next_col in valid_columns:
        next_row = -1
        for r in range(board_height - 1, -1, -1):
            if original_board[r, next_col] == 0:
                next_row = r
                break

        if next_row != -1:
            original_board[next_row, next_col] = opponent_id
            if _static_check_win(original_board, opponent_id, next_row, next_col, game.winning_length):
                return True # Opponent has an immediate threat
            original_board[next_row, next_col] = 0 # Undo simulation

    return False

# --- Main Reward Calculation Function ---

def calculate_reward(game, player_id, row, col, done, reward_type):
    if done:
        if game.winner == player_id:
            # Win + bonus for winning early
            return 10.0 + (game.board_height * game.board_width - np.count_nonzero(game.board)) * 0.05
        elif game.winner is not None:
            return -10.0  # Lost
        else:
            return 0.0  # Draw

    if reward_type != 'shaped':
        return 0.0  # Sparse reward mode

    # --- Intermediate Shaped Rewards ---
    intermediate_reward = 0.0
    opponent_id = 3 - player_id

    if row == -1 or col == -1:
        return 0.0  # No valid move

    # 1. Reward for creating 3-in-a-row (any direction)
    if _check_line_length(game, row, col, player_id, game.winning_length - 1, opponent_id):
        intermediate_reward += 0.3

    # 2. Bonus for diagonal threats
    if _is_diagonal_threat(game, row, col, player_id, game.winning_length - 1):
        intermediate_reward += 1.0

    # 3. Reward for blocking opponent’s 3-in-a-row
    if _check_line_length(game, row, col, player_id, game.winning_length - 1, opponent_id, check_for_player=opponent_id):
        intermediate_reward += 1.5

    # 4. Reward for center control
    center_col = game.board_width // 2
    if col == center_col:
        intermediate_reward += 0.1

    # 5. Penalty for allowing immediate opponent win
    if _check_opponent_immediate_win_threat(game, opponent_id):
        intermediate_reward -= 1.0

    # 6. Reward for creating multiple threats
    potential_wins = 0
    for next_col in game.get_valid_columns():
        for next_row in range(game.board_height - 1, -1, -1):
            if game.board[next_row, next_col] == 0:
                if _check_line_length(game, next_row, next_col, player_id, game.winning_length - 1, opponent_id):
                    potential_wins += 1
                break
    if potential_wins > 1:
        intermediate_reward += 2.0

    # 7. Penalty for stacking without strategic value
    if row < game.board_height - 1 and game.board[row + 1, col] == player_id:
        if not _check_line_length(game, row, col, player_id, 2, opponent_id):
            intermediate_reward -= 0.7

    # 8. Penalty for playing in edge columns
    if col == 0 or col == game.board_width - 1:
        intermediate_reward -= 0.3

    return intermediate_reward