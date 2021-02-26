import random

from interfaces.world import SimWorld
from interfaces.mcts import Mcts
import numpy as np


class Actor:
    def __init__(self, actor_cfg, anet, mcts: Mcts, world: SimWorld):
        self.episodes = actor_cfg["episodes"]
        self.save_episodes = []
        if "num_checkpoints" in actor_cfg and actor_cfg["num_checkpoints"] > 0:
            self.save_episodes = [self.episodes]
            if actor_cfg["num_checkpoints"] > 1:
                self.save_episodes.extend([*range(0, self.episodes, self.episodes / (actor_cfg["num_checkpoints"] - 1))])
        self.anet = anet
        self.mcts: Mcts = mcts
        self.state_manager: SimWorld = world

        self.epsilon = actor_cfg["epsilon"]
        self.epsilon_decay = actor_cfg["epsilon_decay"]
        self.epsilon_min = actor_cfg["epsilon_min"]

    def fit(self):
        wins = 0
        late_wins = 0
        for episode in range(self.episodes):
            print(episode)
            replay_features = []
            replay_targets = []
            if episode in self.save_episodes:
                self.anet.save_params(episode)
            actual_board = self.state_manager.new_state()
            self.mcts.run_root(actual_board)
            while True:
                root_distribution = self.mcts.root_distribution()
                D = self.state_manager.complete_action_dist(root_distribution)
                replay_features.append(self.state_manager.vector(actual_board))
                replay_targets.append(D)

                if random.random() < self.epsilon + self.epsilon_min:
                    action = list(root_distribution.keys())[random.randint(0, len(root_distribution.keys())-1)]
                else:
                    best = float("-inf")
                    for a, p in enumerate(D):
                        if p > best:
                            action = a
                            best = p
                actual_board = self.state_manager.do_action(actual_board, action)
                if self.state_manager.in_end_state(actual_board):
                    break
                self.mcts.run_subtree(actual_board)
            wins += self.state_manager.p1_reward(actual_board)
            self.anet.train(replay_features, replay_targets)
            self.epsilon *= self.epsilon_decay
            if episode > self.episodes / 2:
                late_wins += self.state_manager.p1_reward(actual_board)
        if len(self.save_episodes) > 0:
            self.anet.save_params(self.episodes)
        print("All episodes win percentage: ", wins / self.episodes)
        print("Last 50% episodes win percentage: ", 2 * late_wins / self.episodes)