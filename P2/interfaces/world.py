from typing import List

class SimWorld:

    def new_state(self):
        pass

    def get_actions(self, state, known_not_endstate=False) -> List:
        pass

    def do_action(self, state, action):
        pass

    def in_end_state(self, state) -> bool:
        pass

    def winner(self, state, known_endstate=False):
        pass

    def child_states(self, state):
        return [self.do_action(state, action) for action in self.get_actions(state)]

    def p1_reward(self, state, known_endstate=False):
        pass

    def p1_to_move(self, state):
        pass

    def find_action(self, parent_state, child_state):
        pass

    def visualize(self, states, player_labels=("player1, player2"), use_board_labels=False):
        pass


class AdvancedSimWorld(SimWorld):

    def vector(self, state):
        pass

    def to_array(self, state):
        pass

    def complete_action_dist(self, actions):
        pass

    def action_vector_mask(self, state):
        pass

    def augment_training_data(self, state, action_dist):
        pass