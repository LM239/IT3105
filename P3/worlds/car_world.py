import math
import random
from matplotlib import pyplot as plt

class CarWorld:
    def __init__(self):
        self.x = random.random() / 5 - 0.6
        self.v = 0.0
        self.action_count = 0
        self.episodes = []
        self.episode = []

    def get_actions(self):
        return [-1, 0, 1]

    def coarse_code(self):
        pass

    def get_height(self):
        return math.cos(3 * (self.x + math.pi / 2))

    def do_action(self, a):
        self.episode.append(self.x)
        self.v += 0.001 * a - 0.0025 * math.cos(3 * self.x)
        self.x += self.v
        self.action_count += 1

    def finished(self):
        return self.action_count >= 1000 or self.x == 0.6

    def reset_state(self):
        self.episodes.append(self.episode)
        self.x = random.random() / 5 - 0.6
        self.v = 0.0
        self.action_count = 0
        self.episode = []

    def plot_results(self):
        scatter_y = [len(episode) for episode in self.episodes]
        scatter_x = [*range(len(self.episodes))]
        plt.scatter(scatter_x, scatter_y)

        



