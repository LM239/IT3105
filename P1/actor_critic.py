from collections import defaultdict
import random
from typing import List, Tuple


class ActorCritic:
    
    def __init__(self, actor_config, critic_config, world, episodes):
        self.verify_configs(actor_config, critic_config)

        self.episodes = episodes
        self.actor_lr = actor_config["lr"]
        self.actor_eligibility_decay = actor_config["eligibility_decay"]
        self.actor_discount_factor = actor_config["discount_factor"]
        self.actor_greedy_epsilon = actor_config["greedy_epsilon"]
        self.actor_greedy_epsilon = actor_config["greedy_epsilon"]
        self.actor_epsilon_decay = actor_config["epsilon_decay"]

        self.critic_type = critic_config["type"]
        self.critic_lr = critic_config["lr"]
        self.critic_eligibility_decay = critic_config["eligibility_decay"]
        self.critic_discount_factor = critic_config["discount_factor"]

        if self.critic_type == "neural_net":
            self.critic_size = critic_config["size"]

        self.critic_V = defaultdict(lambda: random.random() * 0.5)
        self.actor_PI = defaultdict(lambda: 0)
        self.world = world

    def fit(self):
        for episode in range(self.episodes):
            self.fit_episode()

    def fit_episode(self):
        actor_eligibility = defaultdict(lambda: 0)
        critic_eligibility = defaultdict(lambda: 0)

        self.world = self.world.reset()
        a = self.use_policy(str(self.world), self.actor_greedy_epsilon)
        state = str(self.world)
        episode = []
        while not self.world.is_end_state_self():
            episode.append((state, str(a)))
            self.world.do_action(a)
            reward = self.world.state_reward_self()
            state_prime = str(self.world)

            a_prime = self.use_policy(state_prime, self.actor_greedy_epsilon)
            if a_prime:
                actor_eligibility[state_prime + str(a_prime)] = 1

            delta = reward + self.critic_discount_factor * self.critic_V[state_prime] - self.critic_V[state]
            critic_eligibility[state] = 1
            for sap in episode:
                state = sap[0]
                action = sap[1]

                self.critic_V[state] += self.critic_lr * delta * critic_eligibility[state]
                critic_eligibility[state] *= self.critic_discount_factor * self.critic_eligibility_decay

                self.actor_PI[state + action] += self.actor_lr * delta * actor_eligibility[state + action]
                actor_eligibility[state + action] *= self.actor_discount_factor * self.actor_eligibility_decay
            state = state_prime
            a = a_prime
            self.actor_greedy_epsilon *= self.actor_epsilon_decay

    def use_policy(self, state: str, epsilon: float) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        actions = self.world.get_actions_self()
        if len(actions) == 0:
            return None
        if random.random() < epsilon:
            return actions[random.randint(0, len(actions) - 1)]
        else:
            best = float('-inf')
            best_actions = []
            for action in actions:
                if self.actor_PI[state + str(action)] >= best:
                    if self.actor_PI[state + str(action)] > best:
                        best = self.actor_PI[state + str(action)]
                        best_actions = [action]
                    else:
                        best_actions.append(action)
            return best_actions[random.randint(0, len(best_actions) - 1)]

    def verify_configs(self, actor_config, critic_config):
        if "lr" not in actor_config:
            print("Missing required actor_config argument: 'lr' \nExiting")
            exit(1)
        if "eligibility_decay" not in actor_config:
            print("Missing required actor_config argument: 'eligibility_decay' \nExiting")
            exit(1)
        if "discount_factor" not in actor_config:
            print("Missing required actor_config argument: 'discount_factor' \nExiting")
            exit(1)
        if "greedy_epsilon" not in actor_config:
            print("Missing required actor_config argument: 'greedy_epsilon' \nExiting")
            exit(1)
        if "greedy_epsilon" not in actor_config:
            print("Missing required actor_config argument: 'greedy_epsilon' \nExiting")
            exit(1)
        if "epsilon_decay" not in actor_config:
            print("Missing required actor_config argument: 'epsilon_decay' \nExiting")
            exit(1)

        if "type" not in critic_config:
            print("Missing required Critic argument: 'type' \nExiting")
            exit(1)
        if not (critic_config["type"] == "neural_net" or critic_config["type"] == "table"):
            print("Unknown critic type: {} \nExiting".format(critic_config["type"]))
            exit(1)
        if critic_config["type"] == "neural_net" and "size" not in critic_config:
            print("Missing required neural net based-Critic argument: 'size' \nExiting")
            exit(1)
        if "lr" not in critic_config:
            print("Missing required Critic argument: 'lr' \nExiting")
            exit(1)
        if "eligibility_decay" not in critic_config:
            print("Missing required Critic argument: 'eligibility_decay' \nExiting")
            exit(1)
        if "discount_factor" not in critic_config:
            print("Missing required Critic argument: 'discount_factor' \nExiting")
            exit(1)


if __name__ == "__main__":
    actor_config = {'lr': 0.1,
                    'eligibility_decay': 0.05,
                    'discount_factor': 0.9,
                    'greedy_epsilon': 0.5,
                    'epsilon_decay': 0.95}
    critic_config = {'type': 'table',
                     #'size': [15, 20, 30, 5, 1],
                     'lr': 0.1,
                     'eligibility_decay': 0.05,
                     'discount_factor': 0.9}
    world_config = {'world': 'peg_solitaire',
                    'type': 'diamond',
                    'size': 4,
                    'display': {
                        'train_display': False,
                        'display_rate': 0.5},
                    }
    from worlds.pegsol_world import PegSolitaire
    world = PegSolitaire(world_config)
    episodes = 100

    actor_critic = ActorCritic(actor_config, critic_config, world, episodes)
    actor_critic.fit()
    world.visualize_episode()
    world.visualize_peg_count()
    exit(0)
