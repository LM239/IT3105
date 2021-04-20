import math
import random

class CarWorld:
    def __init__(self):
        self.state = [random.random() - 0.4, 0.0]

    def get_action(self):
        return [-1, 0, 1]

    def coarse_code(self):
        pass

    def get_height(self):
        return math.cos(3 * (self.state[0] + math.pi / 2))

