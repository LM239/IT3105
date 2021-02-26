from interfaces.world import SimWorld
from interfaces.mcts import Mcts


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

    def fit(self):
        wins = 0
        late_wins = 0
        for episode in range(self.episodes):
            print(episode)
            replay_buffer = []
            if self.anet is not None and episode in self.save_episodes:
                self.anet.save_params()
            actual_board = self.state_manager.new_state()
            action = self.mcts.run_root(actual_board)
            while True:
                D = self.state_manager.complete_action_dist(self.mcts.root_distribution())
                replay_buffer.append((D, actual_board))


                #best = float("-inf")
                #for a, p in enumerate(D): # TODO: epsillon greedy
                    #if p > best:
                        #action = a
                        #best = p
                actual_board = self.state_manager.do_action(actual_board, action)
                if self.state_manager.in_end_state(actual_board):
                    break
                action = self.mcts.run_subtree(actual_board)
            wins += self.state_manager.p1_reward(actual_board)
            if episode > self.episodes / 2:
                late_wins += self.state_manager.p1_reward(actual_board)
        if self.anet is not None and len(self.save_episodes) > 0:
            self.anet.save_params()
        print("All episodes win percentage: ", wins / self.episodes)
        print("Last 50% episodes win percentage: ", 2 * late_wins / self.episodes)