from typing import List

# Interface for the sim world
class SimWorld:

    # Creates a new initial state of the world
    def new_state(self):
        pass

    # Returns the possible actions that can be done in a state according to the rules of the world
    def get_actions(self, state, known_not_endstate=False) -> List:
        pass

    # Performs an action on a state and returns the resulting state
    def do_action(self, state, action):
        pass

    # Checks if the state is an end state
    def in_end_state(self, state) -> bool:
        pass

    # Checks if the state is in an endstate, and if so which player is the winning player
    def winner(self, state, known_endstate=False):
        pass

    # Returns all possible child states for all possible actions in a state
    def child_states(self, state):
        return [self.do_action(state, action) for action in self.get_actions(state)]

    # Returns the reward for player 1
    def p1_reward(self, state, known_endstate=False):
        pass

    # Returns whether it is player 1's turn to move
    def p1_to_move(self, state):
        pass

    # Returns the action that is required to go from a parent state to a child state, if it exists
    def find_action(self, parent_state, child_state):
        pass

    # Creates a progressive visualisation of multiple states
    def visualize(self, states, player_labels=("player1, player2"), use_board_labels=False):
        pass

    def min_depth(self, key):
        pass

class AdvancedSimWorld(SimWorld):

    # Transforms the state to a vector representation
    def vector(self, state):
        pass

    # Returns an array representation of the state for use in neural network
    def to_array(self, state):
        pass


    def complete_action_dist(self, actions):
        pass

    # Returns a mask over the action space where 1 is a possible action, and 0 is an impossible action
    def action_vector_mask(self, state):
        pass

    # Augments the state so we can get more training data
    def augment_training_data(self, state, action_dist):
        pass