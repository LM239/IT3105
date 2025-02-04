from collections import defaultdict
import random
from typing import List, Tuple
from configs.validate_configs import validate_actor_config


class ActorCritic:
    
    def __init__(self, actor_cfg, critic, sim_world, num_episodes, display_episodes):
        validate_actor_config(actor_cfg)  # check all parameters are present

        self.episodes = num_episodes
        self.actor_lr = actor_cfg["lr"]
        self.actor_eligibility_decay = actor_cfg["eligibility_decay"]
        self.actor_discount_factor = actor_cfg["discount_factor"]
        self.actor_greedy_epsilon = actor_cfg["greedy_epsilon"]
        self.actor_epsilon_decay = actor_cfg["epsilon_decay"]
        self.critic = critic  # either neural net or table based

        self.display_episodes = display_episodes  # list of episodes to visualize
        self.actor_PI = defaultdict(lambda: 0)  # defaults to 0 for unknown states
        self.world = sim_world  # world to get available action, reward, etc from

    def fit(self):  # train policy
        for episode_id in range(self.episodes):  # perform episodes
            if episode_id > 0:
                self.world.reset()
            actor_eligibility = defaultdict(lambda: 0)  # defaults to zero for unknown values
            self.critic.reset_eligibilities()  # zero critic eligibilities

            a = self.use_policy(str(self.world.vector()), self.actor_greedy_epsilon)  # current action at s=state
            if not a:
                print("Error: Game has no legal moves\nExiting")
                exit()
            state = self.world.vector()  # vector of board state
            episode = []  # state action pairs for current episodes
            while not self.world.in_end_state():
                episode.append((state, str(a)))
                self.world.do_action(a)  # perform action a
                reward = self.world.state_reward()  # collect reward
                state_prime = self.world.vector()  # get curent state_vector
                a_prime = self.use_policy(str(state_prime), self.actor_greedy_epsilon)  # use policy with epsillon=self.actor_greedy_epsilon
                if a_prime:  # None if state_prime is end state
                    actor_eligibility[str(state_prime) + str(a_prime)] = 1  # update eligibility trace for (state_prime, a_prime)

                delta = self.critic.update(episode, state, state_prime, reward)  # update critic, and recive delta
                for state, action in episode:  # update eligibilities and policy for all (s, a) in the list episode
                    self.actor_PI[str(state) + action] += self.actor_lr * delta * actor_eligibility[str(state) + action]
                    actor_eligibility[str(state) + action] *= self.actor_discount_factor * self.actor_eligibility_decay
                state = state_prime
                a = a_prime
            self.actor_greedy_epsilon *= self.actor_epsilon_decay  # decay epsilon
            self.critic.finish_episode()  # train nn_critic if relevant
            if episode_id in self.display_episodes:
                self.world.visualize_episode()
            print(episode_id)

    def play_episode(self):  # use current policy (epsilon=0)
        self.world = self.world.reset()
        while not self.world.in_end_state():
            self.world.do_action(self.use_policy(str(self.world.vector()), 0))

    def use_policy(self, state: str, epsilon: float) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        actions = self.world.get_actions()  # get possible actions
        if len(actions) == 0:
            return None  # return None when no actions are available (assumed as end state)
        if random.random() < epsilon:
            return actions[random.randint(0, len(actions) - 1)]  # non-greedy choice
        else:  # greedy choice
            best = float('-inf')  # evaluation of best action(s) so far
            best_actions = []  # find best action(s)
            for action in actions:
                if self.actor_PI[state + str(action)] >= best:
                    if self.actor_PI[state + str(action)] > best:
                        best = self.actor_PI[state + str(action)]  # better than all other actions (so far)
                        best_actions = [action]  # update list and evaluation
                    else:
                        best_actions.append(action)  # as good as other action but not better
            return best_actions[random.randint(0, len(best_actions) - 1)]
