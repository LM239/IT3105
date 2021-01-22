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
        delta = reward + self.discount_factor * self.V[str(state_prime)] - self.V[str(state)]
        self.eligibilities[str(state)] = 1
        for state, action in episode:
            self.V[str(state)] += self.lr * delta * self.eligibilities[str(state)]
            self.eligibilities[str(state)] *= self.discount_factor * self.eligibility_decay
        return delta

    def reset_eligibilities(self):
        self.eligibilities = defaultdict(lambda: 0)

    def finish_episode(self):
        pass



