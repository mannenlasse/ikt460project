from copy import deepcopy

def count_winning_moves(game, player_id):
    count = 0
    valid_moves = game.get_valid_columns()
    for move in valid_moves:
        temp_game = deepcopy(game)
        temp_game.current_player = player_id
        result = temp_game.make_move(move)
        if result:
            row, col = result
            if temp_game.winning_moves(row, col):
                count += 1
    return count

def opponent_has_winning_move(game, opponent_id):
    return count_winning_moves(game, opponent_id) >= 1