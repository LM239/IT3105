import random
from typing import Any

from interfaces.world import SimWorld
from interfaces.mcts import Mcts
from interfaces.actornet import ActorNet
import numpy as np


class Actor:
    def __init__(self, anet: ActorNet, world: SimWorld, actor_cfg=None, mcts: Mcts = None):
        self.anet = anet
        self.state_manager: SimWorld = world
        self.mcts: Mcts = mcts
        if actor_cfg is not None:
            self.episodes = actor_cfg["episodes"]
            self.save_episodes = []
            if actor_cfg["num_checkpoints"] > 0:
                self.save_episodes = [self.episodes]
                if actor_cfg["num_checkpoints"] > 1:
                    self.save_episodes.extend([*range(0, self.episodes, int(self.episodes / (actor_cfg["num_checkpoints"] - 1)))])

            self.epsilon = actor_cfg["epsilon"]
            self.epsilon_decay = actor_cfg["epsilon_decay"]
            self.epsilon_min = actor_cfg["epsilon_min"]

    def get_move(self, state: Any) -> int:
        mask = self.state_manager.action_vector_mask(state)
        vector = self.state_manager.vector(state)
        net_out = self.anet.forward(vector)[0]
        masked_out = np.multiply(net_out, mask)
        masked_out = np.divide(masked_out, np.sum(masked_out))
        return np.argmax(masked_out)


    def fit(self):
        wins = 0
        late_wins = 0
        for episode in range(self.episodes):
            print(episode)
            replay_features = []
            replay_targets = []
            if episode in self.save_episodes:
                print("-------------------------------------------------------------------------------------")
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
            if episode >= self.episodes / 2:
                late_wins += self.state_manager.p1_reward(actual_board)
        if len(self.save_episodes) > 0:
            self.anet.save_params(self.episodes)
        print("All episodes win percentage: ", wins / self.episodes)
        print("Last 50% episodes win percentage: ", 2 * late_wins / self.episodes)