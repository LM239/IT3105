from collections import defaultdict
import random
from typing import List, Tuple
from configs.validate_configs import validate_actor_config

class ActorCritic:
    
    def __init__(self, actor_cfg, critic, sim_world, num_episodes, display_episodes):
        validate_actor_config(actor_cfg)

        self.episodes = num_episodes
        self.actor_lr = actor_cfg["lr"]
        self.actor_eligibility_decay = actor_cfg["eligibility_decay"]
        self.actor_discount_factor = actor_cfg["discount_factor"]
        self.actor_greedy_epsilon = actor_cfg["greedy_epsilon"]
        self.actor_epsilon_decay = actor_cfg["epsilon_decay"]
        self.critic = critic

        self.display_episodes = display_episodes
        self.actor_PI = defaultdict(lambda: 0)
        self.world = sim_world

    def fit(self):
        for episode_id in range(self.episodes):
            actor_eligibility = defaultdict(lambda: 0)
            self.critic.reset_eligibilities()

            self.world = self.world.reset()
            a = self.use_policy(str(self.world), self.actor_greedy_epsilon)
            state = self.world.vector()
            episode = []
            while not self.world.is_end_state():
                episode.append((state, str(a)))
                self.world.do_action(a)
                reward = self.world.state_reward()
                state_prime = str(self.world)

                a_prime = self.use_policy(state_prime, self.actor_greedy_epsilon)
                if a_prime:
                    actor_eligibility[state_prime + str(a_prime)] = 1

                delta = self.critic.update(episode, state, state_prime, reward)
                for state, action in episode:
                    self.actor_PI[str(state) + action] += self.actor_lr * delta * actor_eligibility[str(state) + action]
                    actor_eligibility[str(state) + action] *= self.actor_discount_factor * self.actor_eligibility_decay
                state = state_prime
                a = a_prime
            self.actor_greedy_epsilon *= self.actor_epsilon_decay
            self.critic.finish_episode()
            if episode_id in self.display_episodes:
                self.world.visualize_episode()

    def play_episode(self):
        self.world = self.world.reset()
        while not self.world.is_end_state():
            self.world.do_action(self.use_policy(str(self.world), 0))

    def use_policy(self, state: str, epsilon: float) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        actions = self.world.get_actions()
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


if __name__ == "__main__":
    actor_config = {'lr': 0.75,
                    'eligibility_decay': 0.9,
                    'discount_factor': 0.85,
                    'greedy_epsilon': 0.6,
                    'epsilon_decay': 0.85
                    }
    critic_config = {'type': 'table',
                     #'size': [15, 20, 30, 5, 1],
                     'lr': 0.5,
                     'eligibility_decay': 0.9,
                     'discount_factor': 0.85
                     }
    world_config = {'world': 'peg_solitaire',
                    'type': 'triangle',
                    'size': 5,
                    'display_rate': 0.3
                    }
    from worlds.pegsol_world import PegSolitaire
    world = PegSolitaire(world_config)
    episodes = 150

    actor_critic = ActorCritic(actor_config, critic_config, world, episodes)
    actor_critic.fit()
    actor_critic.play_episode()
    #world.visualize_episode()
    world.visualize_peg_count()
    exit(0)
