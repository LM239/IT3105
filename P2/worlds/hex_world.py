

class HexWorld:

    def __init__(self, hex_cfg):
        self.size = hex_cfg["size"]

    def new_world(self):
        return [(1, 0)].extend([(0, 0) for i in self.size ** 2])

    def get_actions(self, state):
        return [i for i, t in enumerate(state[1:]) if t[0] == t[1]]

    def do_action(self, state, action):
        return [reversed(state[0])] + state[1:action] + [state[0]] + state[action + 1:]
