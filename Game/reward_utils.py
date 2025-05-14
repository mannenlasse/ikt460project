import numpy as np

# --- Helper Functions ---

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
    Used for detecting own N-1-in-a-row or blocking opponent's N-1-in-a-row.
    Requires opponent_id to check blocking condition.
    """
    if check_for_player is None:
        check_for_player = player_id

    board = game.board
    height = game.board_height
    width = game.board_width

    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in directions:
        # Removed initial 'count' calculation as it was not used for the return conditions.

        # Recalculate count across the whole line through (row, col) for check_for_player
        line_count = 1 if board[row, col] == check_for_player else 0
        # Positive direction
        for i in range(1, game.winning_length): # Check up to winning_length for full line context
             r, c = row + i * dr, col + i * dc
             if 0 <= r < height and 0 <= c < width and board[r, c] == check_for_player:
                 line_count += 1
             else:
                 break
        # Negative direction
        for i in range(1, game.winning_length): # Check up to winning_length for full line context
             r, c = row - i * dr, col - i * dc
             if 0 <= r < height and 0 <= c < width and board[r, c] == check_for_player:
                 line_count += 1
             else:
                 break

        # Check blocking condition: Did the opponent have length_needed-1 before this move?
        # This means player_id's current move at (row, col) is blocking opponent_id.
        if board[row, col] == player_id and check_for_player == opponent_id:
            # Check count in positive direction *before* the move for opponent_id
            count_pos = 0
            for i in range(1, length_needed): # Check for opponent's line of length_needed-1
                r_opp, c_opp = row + i * dr, col + i * dc
                if 0 <= r_opp < height and 0 <= c_opp < width and board[r_opp, c_opp] == opponent_id:
                    count_pos += 1
                else: break
            # Check count in negative direction *before* the move for opponent_id
            count_neg = 0
            for i in range(1, length_needed): # Check for opponent's line of length_needed-1
                r_opp, c_opp = row - i * dr, col - i * dc
                if 0 <= r_opp < height and 0 <= c_opp < width and board[r_opp, c_opp] == opponent_id:
                    count_neg += 1
                else: break

            # If the sum of opponent's pieces on both sides of the current move spot was length_needed-1
            if count_pos + count_neg >= length_needed - 1:
                 return True # Agent blocked a line of opponent_id that was length_needed-1 long

        # Check own line creation condition
        # This means player_id's current move at (row, col) created a line for player_id.
        elif board[row, col] == player_id and check_for_player == player_id:
             # Avoid double-counting win reward if length_needed is for winning
             if length_needed == game.winning_length:
                  continue # Win is handled by the main reward section
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
        count = 1 # Start with the current piece
        # Count in positive direction
        for i in range(1, length_needed):
            r, c = row + i * dr, col + i * dc
            if 0 <= r < height and 0 <= c < width and board[r, c] == player_id:
                count += 1
            else:
                break
        # Count in negative direction
        for i in range(1, length_needed):
            r, c = row - i * dr, col - i * dc
            if 0 <= r < height and 0 <= c < width and board[r, c] == player_id:
                count += 1
            else:
                break
        if count >= length_needed: # If the total line is at least length_needed
            return True
    return False

def _check_opponent_immediate_win_threat(game, opponent_id):
    """
    Checks if the opponent has a winning move available on their next turn,
    assuming the current board state (after agent's move).
    """
    valid_columns = game.get_valid_columns()
    # Create a true copy for simulation to avoid modifying the actual game board
    sim_board = game.board.copy()
    board_height = game.board_height

    for next_col in valid_columns:
        next_row = -1
        for r in range(board_height - 1, -1, -1):
            if sim_board[r, next_col] == 0:
                next_row = r
                break

        if next_row != -1:
            sim_board[next_row, next_col] = opponent_id # Simulate opponent's move
            if _static_check_win(sim_board, opponent_id, next_row, next_col, game.winning_length):
                return True # Opponent has an immediate threat
            sim_board[next_row, next_col] = 0 # Undo simulation for next check

    return False

# --- Main Reward Calculation Function ---

def calculate_reward(game, player_id, row, col, done, reward_type):
    if done:
        if game.winner == player_id:
            # Significantly increased win reward + smaller bonus for winning early
            return 50.0 + (game.board_height * game.board_width - np.count_nonzero(game.board)) * 0.1
        elif game.winner is not None:
            return -50.0  # Significantly increased loss penalty
        else:
            return 10.0  # Draw - Significantly increased from 0.5

    if reward_type != 'shaped':
        return 0.0  # Sparse reward mode

    # --- Intermediate Shaped Rewards ---
    intermediate_reward = 0.0
    opponent_id = 3 - player_id

    # CRITICAL: Penalty for allowing opponent an immediate win if they play next (after current player's move)
    # This check is more about the state *after* the current move.
    if _check_opponent_immediate_win_threat(game, opponent_id): # Corrected: removed WIN_LENGTH argument
         intermediate_reward -= 30.0 # Increased penalty (was -25.0)

    # CRITICAL: Reward for blocking opponent's WINNING threat (game.winning_length - 1 pieces)
    # This means player_id's current move at (row, col) blocks opponent_id from completing a line of game.winning_length.
    # The _check_line_length call for this uses length_needed = game.winning_length.
    # Inside _check_line_length, it checks if opponent had (length_needed - 1) pieces.
    if _check_line_length(game, row, col, player_id, game.winning_length, opponent_id, check_for_player=opponent_id):
        intermediate_reward += 25.0  # Increased reward (was 15.0) for blocking a direct N-1 win threat

    # CRITICAL: Reward for creating own (game.winning_length - 1)-in-a-row (e.g., 3-in-a-row for Connect 4)
    # This applies to all directions.
    # The _check_line_length call for this uses length_needed = game.winning_length - 1.
    if not game.winner and _check_line_length(game, row, col, player_id, game.winning_length - 1, opponent_id, check_for_player=player_id):
        intermediate_reward += 20.0 # Significantly increased reward (was 5.0)

    # Reward for making a line of 2 (less critical)
    if not game.winner and _check_line_length(game, row, col, player_id, 2, opponent_id, check_for_player=player_id): # Corrected call
        intermediate_reward += 0.5 # Keep this small

    # Reward for blocking opponent's line of 2 (less critical)
    # To block a 2-in-a-row, opponent would have 2 pieces, current move makes it 3 for them if not blocked.
    # So, length_needed for _check_line_length (when checking opponent) should be 2 + 1 = 3.
    if _check_line_length(game, row, col, player_id, 3, opponent_id, check_for_player=opponent_id):
        intermediate_reward += 3.0 # Slightly increased (was 1.0 or 2.0)

    # Bonus for diagonal threats of (winning_length - 1) - can be an add-on
    if not game.winner and _is_diagonal_threat(game, row, col, player_id, game.winning_length - 1):
        intermediate_reward += 5.0 # Was 7.0, adjusted as main N-1 reward is now higher

    # Center control
    center_col_low = (game.board_width -1) // 2
    center_col_high = game.board_width // 2
    if col == center_col_low or col == center_col_high :
        intermediate_reward += 0.5 # Reduced, as critical plays are more important (was 1.0)

    # Penalty for stacking on own piece without creating at least a 2-in-a-row
    if row < game.board_height - 1 and game.board[row + 1, col] == player_id:
        if not _check_line_length(game, row, col, player_id, 2, opponent_id, check_for_player=player_id):
            intermediate_reward -= 2.0 # Was -3.0

    # Penalty for playing in edge columns (unless it's a critical move)
    is_critical_offensive_move = not game.winner and _check_line_length(game, row, col, player_id, game.winning_length - 1, opponent_id, check_for_player=player_id)
    is_critical_defensive_move = _check_line_length(game, row, col, player_id, game.winning_length, opponent_id, check_for_player=opponent_id)

    if (col == 0 or col == game.board_width - 1) and not (is_critical_offensive_move or is_critical_defensive_move):
        intermediate_reward -= 1.0 # Was -1.5

    # The "potential_future_threats" logic (reward #6 in your file) is complex and its current
    # implementation with temporary board modification is risky.
    # For now, I'm commenting it out to focus on the immediate impact of the current move.
    # If this is desired, it should be revisited with a safer implementation.
    # Original section for potential_future_threats:
    # potential_future_threats = 0
    # board_after_current_move = game.board.copy()
    # for next_col_sim in game.get_valid_columns():
    #     next_row_sim = -1
    #     for r_sim in range(game.board_height - 1, -1, -1):
    #         if board_after_current_move[r_sim, next_col_sim] == 0:
    #             next_row_sim = r_sim
    #             break
    #     if next_row_sim != -1:
    #         original_game_board_ref = game.board
    #         game.board = board_after_current_move # Risky
    #         board_after_current_move[next_row_sim, next_col_sim] = player_id
    #         if _check_line_length(game, next_row_sim, next_col_sim, player_id, game.winning_length - 1, opponent_id, check_for_player=player_id):
    #             potential_future_threats += 1
    #         game.board = original_game_board_ref # Restore
    #         board_after_current_move[next_row_sim, next_col_sim] = 0
    # if potential_future_threats > 1:
    #     intermediate_reward += 10.0

    # Ensure row/col are valid before returning, though this should be caught earlier.
    if row == -1 or col == -1: # Should not happen if move is valid before calling this
        return -100.0 # Heavy penalty for an invalid state passed

    return intermediate_reward