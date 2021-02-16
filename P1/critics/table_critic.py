import random
from collections import defaultdict
from configs.validate_configs import validate_critic_config


class TableCritic:

    def __init__(self, critic_cfg):
        validate_critic_config(critic_cfg)  # validate config
        self.eligibilities = defaultdict(lambda: 0)  #  defaults to 0 for unknown states
        self.V = defaultdict(lambda: random.random() * 0.5)  #  defaults to random small value for unknown states

        self.lr = critic_cfg["lr"]
        self.eligibility_decay = critic_cfg["eligibility_decay"]
        self.discount_factor = critic_cfg["discount_factor"]

    def update(self, episode, state, state_prime, reward):
        delta = reward + self.discount_factor * self.V[str(state_prime)] - self.V[str(state)]  # find delta
        self.eligibilities[str(state)] = 1  # update eligibility of state
        for state, action in episode:  # update Value and eligibilities for all (s, a)
            self.V[str(state)] += self.lr * delta * self.eligibilities[str(state)]
            self.eligibilities[str(state)] *= self.discount_factor * self.eligibility_decay
        return delta

    def reset_eligibilities(self):
        self.eligibilities = defaultdict(lambda: 0) # reset eligibilities (used before each episode)

    def finish_episode(self):
        # Does not need special end of episode handling
        pass



