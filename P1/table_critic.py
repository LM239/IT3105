import random
from collections import defaultdict
from configs.validate_configs import validate_critic_config

class TableCritic:

    def __init__(self, critic_cfg):
        validate_critic_config(critic_cfg)
        self.eligibilities = defaultdict(lambda: 0)
        self.V = defaultdict(lambda: random.random() * 0.5)

        self.lr = critic_cfg["lr"]
        self.eligibility_decay = critic_cfg["eligibility_decay"]
        self.discount_factor = critic_cfg["discount_factor"]

    def update(self, episode, state, state_prime, reward):
        delta = reward + self.discount_factor * self.V[state_prime] - self.V[state]
        self.eligibilities[state] = 1
        for state, action in episode:
            self.V[state] += self.lr * delta * self.eligibilities[state]
            self.eligibilities[state] *= self.discount_factor * self.eligibility_decay
        return delta

    def reset_eligibilities(self):
        self.eligibilities = defaultdict(lambda: 0)



