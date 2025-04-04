class Agent:
    def __init__(self, player_id):
        self.player_id = player_id

    def select_action(self):
        raise NotImplementedError("base agent: Must be implemented by subclass.")