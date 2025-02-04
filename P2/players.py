from interfaces.actornet import ActorNet
from interfaces.world import AdvancedSimWorld
import numpy as np


class Player:
    def __init__(self, anet, sim_world, name):
        self.name = name
        self.anet: ActorNet = anet
        self.state_manager: AdvancedSimWorld = sim_world

    def get_action_dist(self, state):
        mask = self.state_manager.action_vector_mask(state)
        vector = self.state_manager.to_array(state)
        net_out = self.anet.forward(vector)[0]
        return np.multiply(net_out, mask)  # mask net output for legal acition distribution

    def play(self, state):
        pass


class GreedyPlayer(Player):
    def __init__(self, anet, sim_world, name="Un-named player"):
        super().__init__(anet, sim_world, name)

    def play(self, state):
        masked_out = super().get_action_dist(state)
        masked_out = np.divide(masked_out, np.sum(masked_out))
        return np.argmax(masked_out)  # greedy action selection


class ProbabilisticPlayer(Player):
    def __init__(self, anet, sim_world, name="Un-named player"):
        super().__init__(anet, sim_world, name)

    def play(self, state):
        masked_out = super().get_action_dist(state) ** 2
        masked_out = np.divide(masked_out, np.sum(masked_out))
        return np.random.choice(np.arange(len(masked_out)), p=masked_out)  # use probability dist for action selection


class HumanPlayer:
    def __init__(self, sim_world):
        self.name = input("What is your 1337gamer-tag?")
        self.state_manager = sim_world

    def play(self, state):
        player_tags = (self.name, "Terminator") if self.state_manager.p1_to_move(state) else ("Terminator", self.name)
        self.state_manager.visualize([state], player_tags, True)
        actions = self.state_manager.get_actions(state)
        while True:
            action: str = input("Choose one valid action: {}".format(actions))
            if action.isnumeric() and int(action) in actions:
                break
        return int(action)


