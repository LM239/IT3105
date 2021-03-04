import random
from topp import compete
from players import ProbabilisticPlayer

from interfaces.world import AdvancedSimWorld
from interfaces.mcts import Mcts
from interfaces.actornet import ActorNet
import numpy as np


class Actor:
    def __init__(self, anet: ActorNet, world: AdvancedSimWorld, actor_cfg=None, mcts: Mcts = None):
        self.anet = anet
        self.state_manager: AdvancedSimWorld = world
        self.mcts: Mcts = mcts
        self.save_dir = actor_cfg["file_structure"]
        self.competition_games = actor_cfg["competition_games"]
        self.train_ex_size = actor_cfg["train_ex_size"]
        self.episodes = actor_cfg["episodes"]
        self.epsilon = actor_cfg["epsilon"]
        self.epsilon_min = actor_cfg["epsilon_min"]
        self.epsilon_decay = actor_cfg["epsilon_decay"]

        self.save_episodes = []
        if actor_cfg["num_checkpoints"] > 0:
            self.save_episodes = [self.episodes]
            if actor_cfg["num_checkpoints"] > 1:
                self.save_episodes.extend([*range(0, self.episodes, int(self.episodes / (actor_cfg["num_checkpoints"] - 1)))])

    def generate_examples(self):
        replay_features = []
        replay_targets = []
        actual_board = self.state_manager.new_state()
        self.mcts.run_root(actual_board)
        while len(replay_targets) < self.train_ex_size:
            print("Currently on {} of {} training examples".format(len(replay_targets), self.train_ex_size), end="\r")
            root_distribution: dict = self.mcts.root_distribution()
            D = self.state_manager.complete_action_dist(root_distribution)
            augmented_boards, augmented_Ds = self.state_manager.augment_training_data(self.state_manager.to_array(actual_board), D)
            replay_features.extend(augmented_boards)
            replay_targets.extend(augmented_Ds)

            if random.random() < self.epsilon + self.epsilon_min:
                action = list(root_distribution.keys())[random.randint(0, len(root_distribution.keys())-1)]
            else:
                action = np.random.choice(list(root_distribution.keys()), p=list(root_distribution.values()))
            actual_board = self.state_manager.do_action(actual_board, action)
            if self.state_manager.in_end_state(actual_board):
                actual_board = self.state_manager.new_state()
            self.mcts.run_subtree(actual_board)
        print("\n")
        return replay_features, replay_targets

    def fit(self):
        for episode in range(self.episodes):
            print(episode)
            train_features, train_targets = self.generate_examples()
            if episode in self.save_episodes:
                self.anet.save_params(self.save_dir, "checkpoint_" + str(episode) + ".h5")

            self.anet.save_params(self.save_dir, "temp.h5")
            self.anet.train(train_features, train_targets)

            untrained_competitor = ProbabilisticPlayer(self.anet.__class__(model_file=(self.save_dir + "temp.h5")), self.state_manager)
            trained_competitor = ProbabilisticPlayer(self.anet, self.state_manager)
            print("Running competition with {} games".format(self.competition_games))
            trained_wins, untrained_wins = compete(trained_competitor, untrained_competitor, self.competition_games, self.state_manager)

            if trained_wins < untrained_wins:
                self.anet.load_params(self.save_dir + "temp.h5")
            print("New model won {} of {} games ({}%)".format(trained_wins, self.competition_games, trained_wins * 100 / self.competition_games))
            self.epsilon *= self.epsilon_decay

        if len(self.save_episodes) > 0:
            self.anet.save_params(self.save_dir, "checkpoint_" + str(self.episodes) + ".h5")
        self.anet.save_params(self.save_dir, "best.h5")
