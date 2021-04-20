import math
import random
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np

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

    def get_height(self, x):
        return math.cos(3 * (x + math.pi / 2))

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
        scatter_y = np.array([len(episode) for episode in self.episodes])
        scatter_x = np.arange(len(self.episodes))
        plt.scatter(scatter_x, scatter_y)

        plt.show()

        anim_ep = self.episodes[-1]
        fig, ax = plt.subplots()

        x_range = np.arange(-1.2, 0.6, 0.001)
        mountain = np.cos(3 * (x_range + math.pi / 2))
        m_plot = plt.plot(x_range, mountain)

        ax = plt.axis([-1.2, 0.6, -1.5, 1.5])

        car, = plt.plot([anim_ep[0]], [self.get_height(anim_ep[0])], 'ro')

        def animate(i):
            y = math.cos(3 * (i + math.pi / 2))
            car.set_data(i, y)
            return car,

        # create animation using the animate() function
        anim = animation.FuncAnimation(fig, animate, frames=anim_ep, interval=20, blit=True, repeat=True)

        plt.show()





