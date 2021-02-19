from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

from world import SimWorld

class NimWorld(SimWorld):

    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k

    def new_state(self):
        return self.n, (1, 0)

    def get_actions(self, state):
        return [*range(1,self.k+1)]

    def do_action(self, state, action):
        return state[0]-action, tuple(reversed(state[1]))

    def in_end_state(self, state) -> bool:
        return state[0] <= 0

    def visualize(self, states):
        plt.plot(states[:][0])
        plt.show()

    def vector(self, state):
        return state[0], state[1][0], state[1][1]


if __name__ == "__main__":
    world = NimWorld(10, 3)
    state = world.new_state()
    while(not world.in_end_state(state)):
        print(world.vector(state))
        actions = world.get_actions(state)
        action = int(input("Choose action: " + str(actions)))
        state = world.do_action(state, action)
    print("Winner is:", state[1])