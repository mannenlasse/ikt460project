import numpy as np


def _static_check_win(board, player_id, row, col, win_length):
    
    #Static check if a piece on boardresults in a win. 
    
    height, width = board.shape
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  

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
 
    if check_for_player is None:
        check_for_player = player_id

    board = game.board
    height = game.board_height  
    width = game.board_width  

    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in directions:
        count = 1
        if board[row, col] != check_for_player:
             count = 0

        # Count in positive direction
        for i in range(1, length_needed + 1): 
            r, c = row + i * dr, col + i * dc
            if 0 <= r < height and 0 <= c < width and board[r, c] == check_for_player:
                count += 1
            else:
                break
        # Count in negative direction
        for i in range(1, length_needed + 1):  
            r, c = row - i * dr, col - i * dc
            if 0 <= r < height and 0 <= c < width and board[r, c] == check_for_player:
                count += 1
            else:
                break

 
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


        # Check blocking condition:  opponent have almost win before this move?
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

                 return True 

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

def _check_opponent_immediate_win_threat(game, opponent_id):
 
    valid_columns = game.get_valid_columns()
    original_board = game.board.copy()
    board_height = game.board_height 

    for next_col in valid_columns:
        next_row = -1
        for r in range(board_height - 1, -1, -1):
            if original_board[r, next_col] == 0:
                next_row = r
                break

        if next_row != -1:
            original_board[next_row, next_col] = opponent_id
            if _static_check_win(original_board, opponent_id, next_row, next_col, game.winning_length):
                return True 
            original_board[next_row, next_col] = 0 

    return False

# Main Reward Calculation Function

def calculate_reward(game, player_id, row, col, done, reward_type):
 
    # --- Terminal Rewards ---
    if done:
        if game.winner == player_id:
            return 10.0  # Win
        elif game.winner is not None: # Opponent won
            return -10.0 # Loss
        else: # Draw
            return 0.0

    # Intermediate Rewards 
    if reward_type == 'shaped':
        intermediate_reward = 0.0
        opponent_id = 3 - player_id  

        # Check if row/col are valid  
        if row == -1 or col == -1:
             return 0.0 

        # Reward for creating 3-in-a-row  
        if _check_line_length(game, row, col, player_id, game.winning_length - 1, opponent_id):
            intermediate_reward += 0.5

        # Reward for blocking opponent's 3-in-a-row
        if _check_line_length(game, row, col, player_id, game.winning_length - 1, opponent_id, check_for_player=opponent_id):
             intermediate_reward += 0.8




        #  Reward for playing in the center column
        center_col = game.board_width // 2
        if col == center_col:
            intermediate_reward += 0.1

        # Penalty for allowing opponent an immediate win next turn
        if _check_opponent_immediate_win_threat(game, opponent_id):
            intermediate_reward -= 1.0

        if col == 0:
            intermediate_reward -= 0.05

        return intermediate_reward 
    else:
        #   return 0 for non-terminal states
        return 0.0