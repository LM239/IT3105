import math
import random

class CarWorld:
    def __init__(self):
        self.state = [random.random() / 5 - 0.6, 0.0]
        self.action_count = 0

    def get_action(self):
        return [-1, 0, 1]

    def coarse_code(self):
        pass

    def get_height(self):
        return math.cos(3 * (self.state[0] + math.pi / 2))

    def do_action(self, a):
        self.state[1] += 0.001 * a - 0.0025 * math.cos(3 * self.state[0])
        self.state[0] += self.state[1]
        self.action_count += 1

    def halt_episode(self):
        return self.action_count >= 1000 or self.state[0] == 0.6

    def reset_state(self):
        self.state = [random.random() / 5 - 0.6, 0.0]
        self.action_count = 0



