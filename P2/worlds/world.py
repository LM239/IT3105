from typing import List


class SimWorld:

    def new_state(self):
        pass

    def get_actions(self, state) -> List:
        pass

    def do_action(self, state, action):
        pass

    def in_end_state(self, state) -> bool:
        pass

    def visualize(self, states):
        pass

    def vector(self, state):
        pass

    def winner(self, state):
        pass

    def child_states(self, state):
        return [self.do_action(state, action) for action in self.get_actions(state)]
