import random
from topp import compete
from players import ProbabilisticPlayer
from interfaces.world import AdvancedSimWorld
from interfaces.mcts import Mcts
from interfaces.actornet import ActorNet
import numpy as np
import pickle
import glob
import os


class TourActor:
    def __init__(self, anet: ActorNet, world: AdvancedSimWorld, actor_cfg=None, mcts: Mcts = None):
        self.anet = anet
        self.state_manager: AdvancedSimWorld = world
        self.mcts: Mcts = mcts
        self.anet_dir = actor_cfg["anet_dir"]
        self.competition_games = actor_cfg["competition_games"]
        self.train_games = actor_cfg["train_games"]
        self.episodes = actor_cfg["episodes"]
        self.epsilon = actor_cfg["epsilon"]
        self.epsilon_min = actor_cfg["epsilon_min"]
        self.epsilon_decay = actor_cfg["epsilon_decay"]
        self.display_games = actor_cfg["display_games"]
        self.save_data = actor_cfg["save_data"]
        self.data_dir = actor_cfg["data_dir"] if self.save_data else None
        self.win_margin = actor_cfg["win_margin"]
        self.save_episodes = []
        if actor_cfg["num_checkpoints"] > 0:
            self.save_episodes = [self.episodes]
            if actor_cfg["num_checkpoints"] > 1:
                self.save_episodes.extend([*range(0, self.episodes, int(self.episodes / (actor_cfg["num_checkpoints"] - 1)))])

    def generate_examples(self, episode):
        replay_features = []
        replay_targets = []
        games = 0
        actual_board = self.state_manager.new_state()
        states = [actual_board]
        extended_searches = 0
        visits, extended = self.mcts.run_root(actual_board)
        while games < self.train_games:
            extended_searches += extended
            root_distribution: dict = self.mcts.root_distribution()
            D = self.state_manager.complete_action_dist(root_distribution)
            replay_targets.append(D)

            print(f"Currently on {len(replay_targets)} training examples, {games} of {self.train_games} games, and {visits} rollouts with {extended_searches} extended searches            ", end="\r")

            if random.random() < self.epsilon + self.epsilon_min:
                action = list(root_distribution.keys())[random.randint(0, len(root_distribution.keys())-1)]
            else:
                action = np.random.choice(list(root_distribution.keys()), p=list(root_distribution.values()))
            actual_board = self.state_manager.do_action(actual_board, action)
            states.append(actual_board)
            if self.state_manager.in_end_state(actual_board):
                if games + self.train_games * episode in self.display_games:
                    self.state_manager.visualize(states)
                games += 1
                replay_features.extend(states[:-1])
                actual_board = self.state_manager.new_state()
                states = [actual_board]
                visits, extended = self.mcts.run_root(actual_board, True)
                continue
            visits, extended = self.mcts.run_subtree(actual_board)
        print("\n")
        return replay_features, replay_targets

    def fit(self):
        for episode in range(self.episodes):
            print(episode)
            states, action_visits = self.generate_examples(episode)
            if self.save_data:
                file_name = "eps_" + str(self.epsilon_min) + "_" + str(self.train_games) + "games_" + str(episode) + ".p"
                os.makedirs(os.path.dirname(self.data_dir + file_name), exist_ok=True)
                with open(self.data_dir + file_name, "wb") as file:
                    pickle.dump((states, action_visits), file)


            augmented_features = []
            augmented_targets = []
            for feature, target in zip(states, action_visits):
                augmented_boards, augmented_Ds = self.state_manager.augment_training_data(self.state_manager.to_array(feature), target)
                augmented_features.extend(augmented_boards)
                augmented_targets.extend(augmented_Ds)

            if episode in self.save_episodes:
                self.anet.save_params(self.anet_dir, "checkpoint_" + str(episode))

            self.anet.save_params(self.anet_dir, "temp")
            self.anet.train(augmented_features, augmented_targets)

            untrained_competitor = ProbabilisticPlayer(self.anet.__class__(model_file=(self.anet_dir + "temp")), self.state_manager)
            trained_competitor = ProbabilisticPlayer(self.anet, self.state_manager)
            print("Running competition with {} games".format(self.competition_games))
            trained_wins, untrained_wins = compete(trained_competitor, untrained_competitor, self.competition_games, self.state_manager)

            if trained_wins < self.competition_games // 2 + self.win_margin:
                self.anet.load_params(self.anet_dir + "temp")
            print("New model won {} of {} games ({}%)".format(trained_wins, self.competition_games, trained_wins * 100 / self.competition_games if self.competition_games > 0 else 0))

            self.anet.save_params(self.anet_dir, "best")
            self.epsilon *= self.epsilon_decay

        if len(self.save_episodes) > 0:
            self.anet.save_params(self.anet_dir, "checkpoint_" + str(self.episodes))

    def fit_data(self):
        files = glob.glob(self.data_dir + "*.p")
        episode = 0
        for file in files:
            print("Loading data from", file)
            states, action_visits = pickle.load(open(file, "rb"))
            augmented_features = []
            augmented_targets = []
            if (len(states) > len(action_visits)):
                states = [state for state in states if not self.state_manager.in_end_state(state)]
                print(len(states), len(action_visits))
                with open(file, "wb") as file:
                    pickle.dump((states, action_visits), file)
                    exit(0)

            for feature, target in zip(states, action_visits):
                augmented_boards, augmented_Ds = self.state_manager.augment_training_data(self.state_manager.to_array(feature), target)
                augmented_features.extend(augmented_boards)
                augmented_targets.extend(augmented_Ds)
            if episode in self.save_episodes:
                self.anet.save_params(self.anet_dir, "checkpoint_" + str(episode))

            self.anet.save_params(self.anet_dir, "temp")
            self.anet.train(augmented_features, augmented_targets)

            untrained_competitor = ProbabilisticPlayer(self.anet.__class__(model_file=(self.anet_dir + "temp")), self.state_manager)
            trained_competitor = ProbabilisticPlayer(self.anet, self.state_manager)
            print("Running competition with {} games".format(self.competition_games))
            trained_wins, untrained_wins = compete(trained_competitor, untrained_competitor, self.competition_games, self.state_manager)

            if trained_wins < untrained_wins:
                self.anet.load_params(self.anet_dir + "temp")
            print("New model won {} of {} games ({}%)".format(trained_wins, self.competition_games, trained_wins * 100 / self.competition_games if self.competition_games > 0 else 0))

            self.anet.save_params(self.anet_dir, "best")
            episode += 1