from worlds.world import SimWorld
from mcts import Mcts


class Actor:
    def __init__(self, actor_cfg, anet, mcts: Mcts, world: SimWorld):
        self.episodes = actor_cfg["episodes"]
        self.save_episodes = []
        if "num_checkpoints" in actor_cfg:
            self.save_episodes = [self.episodes]
            if actor_cfg["num_checkpoints"] > 1:
                self.save_episodes.extend([*range(0, self.episodes, self.episodes / (actor_cfg["num_checkpoints"] - 1))])
        self.anet = anet
        self.mcts: Mcts = mcts
        self.state_manager: SimWorld = world

    def fit(self):
        for episode in self.episodes:
            replay_buffer = []
            if self.anet is not None and episode in self.save_episodes:
                self.anet.save_params()
            actual_board = self.state_manager.new_state()
            self.mcts.run_root(actual_board)
            while not self.state_manager.in_end_state(actual_board):
                D = self.state_manager.action_vector(self.mcts.root_distribution())
                replay_buffer.append((D, actual_board))

                if self.anet is not None:
                    #D = argmax a: D
                    pass
                else:
                    action = self.mcts.default_policy(actual_board)
                actual_board = self.state_manager.do_action(actual_board, action)
                self.mcts.run_subtree(actual_board)

        if self.anet is not None and len(self.save_episodes) > 0:
            self.anet.save_params()