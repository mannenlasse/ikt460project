class Agent:
    def __init__(self, current_player):
        self.current_player = current_player



    def select_action(self, game):
        raise NotImplementedError("base agent: Must be implemented by subclass.")