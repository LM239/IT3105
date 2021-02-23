from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from configs.validate_configs import validate_nim

from worlds.world import SimWorld


class NimWorld(SimWorld):

    def __init__(self, world_config):
        validate_nim(world_config)
        self.n = world_config["n"]
        self.k = world_config["k"]

    def new_state(self):
        return self.n, (1, 0)

    def get_actions(self, state):
        return [*range(1, min(self.k, state[0])+1)]

    def do_action(self, state, action):
        return state[0] - action, tuple(reversed(state[1]))

    def in_end_state(self, state) -> bool:
        return state[0] <= 0

    def visualize(self, states):
        plt.plot(states[:][0])
        plt.show()

    def winner(self, state):
        return tuple(reversed(state[1])) if self.in_end_state(state) else None

    def vector(self, state):
        return state[0], state[1][0], state[1][1]

    def p1_reward(self, state):
        return state[1][1] if self.in_end_state(state) else None

    def p1_to_move(self, state):
        return state[1][0] == 1

    def find_action(self, parent_state, child_state):
        return parent_state[0] - child_state[0]


if __name__ == "__main__":
    world = NimWorld({"n":10,"k":3})
    state = world.new_state()
    while not world.in_end_state(state):
        print(world.vector(state))
        actions = world.get_actions(state)
        action = int(input("Choose action: " + str(actions)))
        state = world.do_action(state, action)
    print("Winner is:", state[1])