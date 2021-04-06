import time
import sys
import numpy as np
from typing import List, Any
from collections import defaultdict
from interfaces.world import SimWorld
from search.treesearch import default_search
from interfaces.Node import Node
from interfaces.mcts import Mcts
from interfaces.actornet import ActorNet
from math import sqrt, log
import gc
import pickle
import os


def dd():
    return defaultdict(dd2)


def dd2():
    return 0.5


class McRave(Mcts):

    def __init__(self, mcts_cfg, state_manager, anet, node_search=default_search):
        self.biases = [[0, 0]]
        self.Q = defaultdict(dd)
        self.amaf_Q = defaultdict(dd)
        self.state_manager: SimWorld = state_manager

        if "q_dir" in mcts_cfg:  # load previous Q dicts as heuristic
            try:
                self.amaf_Q = pickle.load(open(mcts_cfg["q_dir"] + "amaf_q.p", "rb"))
                self.Q = pickle.load(open(mcts_cfg["q_dir"] + "q.p", "rb"))
                print("Loaded amaf_q and Q dict with {} keys, and {} keys, respectively ({} bytes (not accurate))".format(len(self.amaf_Q), len(self.Q), sys.getsizeof(self.amaf_Q) + sys.getsizeof(self.Q)))
                self.reset_bias()
                print("using biases ", self.biases)
            except FileNotFoundError:
                print("Could not load q_dicts")
                pass

        self.search_games = mcts_cfg["search_games"]
        self.min_confidence = mcts_cfg["h_confidence"]
        self.search_duration = mcts_cfg["search_duration"]
        self.max_rollouts = mcts_cfg["max_rollouts"]
        self.root: Node | None = None
        self.og_root: Node | None = None
        self.node_search = node_search
        self.c = mcts_cfg["c"]
        self.anet: ActorNet = anet

    def save_Qs(self, dir):  # save Q dicts for use as heuristic
        os.makedirs(os.path.dirname(dir + "q_dicts/amaf_q.p"), exist_ok=True)
        with open(dir + "q_dicts/amaf_q.p", "wb") as file:
            pickle.dump(self.amaf_Q, file)
        os.makedirs(os.path.dirname(dir + "q_dicts/q.p"), exist_ok=True)
        with open(dir + "q_dicts/q.p", "wb") as file:
            pickle.dump(self.Q, file)

    def update_bias(self, state: str, action: int, level: int):
        new_bias = abs(self.Q[state][action] - self.amaf_Q[state][action])
        if level == len(self.biases):
            self.biases.append([new_bias, 1])
        else:
            prev_bias, count = self.biases[level]
            if count < 2 ** 16:  # max count
                count += 1
            prev_bias += (new_bias - prev_bias) / count  # change average
            self.biases[level] = [prev_bias, count]

    def reset_bias(self):  # reset bias
        self.biases = [[bias, min(count, 100)] for bias, count in self.biases]
        # get bias for each available depth in Q and amaf_Q
        for state in self.Q.keys():
            level = self.state_manager.min_depth(state)
            if level >= len(self.biases):
                self.biases.extend([[0, 0] for i in range(level - len(self.biases) + 1)])
            for action in self.Q[state].keys():
                self.update_bias(state, action, level)

    def run_root(self, state: Any, use_og_root=False):
        if len(self.amaf_Q.keys()) + len(self.Q.keys()) > 6.0 * 10**6:
            self.reset_bias()
            self.prune_dicts()
        if use_og_root:
            self.root = self.og_root  # retain root
        else:
            del self.root
            del self.og_root
            print("\ngc freed {} objects".format(gc.collect()))  # free some memory
            self.root = self.new_node(state, 0)
            self.og_root = self.root
        return self.simulate(self.root.state, self.search_duration, self.search_games)  # simulate from root state

    def prune_dicts(self):
        print("\nPruning dicts")
        new_amaf_dict = defaultdict(dd)  # create new smaller dicts
        for key in self.amaf_Q.keys():
            if self.state_manager.min_depth(key) >= 29:
                new_amaf_dict[key] = self.amaf_Q[key]
        del self.amaf_Q
        self.amaf_Q = new_amaf_dict
        new_q_dict = defaultdict(dd)
        for key in self.Q.keys():
            if self.state_manager.min_depth(key) >= 29:
                new_q_dict[key] = self.Q[key]
        del self.Q
        self.Q = new_q_dict
        print("New amaf_q and Q dicts have {} keys, and {} keys, respectively ({} bytes (not accurate))".format(
            len(self.amaf_Q), len(self.Q), sys.getsizeof(self.amaf_Q) + sys.getsizeof(self.Q)))

    def run_subtree(self, state: Any):
        # run from new subtree (one of current root's children)
        action = self.state_manager.find_action(self.root.state, state)
        if action in self.root.child_actions:
            self.root = self.root.children[self.root.child_actions.index(action)]
        else:
            self.root = self.new_node(state, self.root.level + 1)
        return self.simulate(self.root.state, self.search_duration, self.search_games)

    def simulate(self, state, search_duration, num_searches, extended=False):
        rollouts = 0
        now = time.time()
        while True:
            nodes, actions = self.tree_search(state)
            rollout_actions, z = self.rollout_sim(nodes[-1].state)
            self.backup(nodes, actions + rollout_actions, z)
            rollouts += 1
            if time.time() - now > search_duration: # perform rollouts until search_duration exceeded
                if rollouts >= num_searches:
                    break
                now = time.time() - 3 * search_duration / 4  #  if number of searches below minimum reshhold, continue with less time until next check
            if rollouts >= self.max_rollouts: # if exceeding max_rollouts terminate
                break
        #  if most visited action does not have best Q, extend search (can only be done once)
        if not (extended or self.best_policy_action(self.root) == self.most_visited_child_action(self.root)):
            return rollouts + self.simulate(state, search_duration / 2, int(num_searches / 2), True)[0], 1
        return rollouts, 0

    def tree_search(self, state):
        node: Node = self.root
        actions: List[int] = []
        nodes: List[Node] = []
        while True:  # Stops if game is over
            if node is None:  # state does not exist in tree: add state and return
                node = self.new_node(state, nodes[-1].level + 1, self.state_manager.in_end_state(state))
                self.insert_node(nodes[-1], node, actions[-1])
                nodes.append(node)
                return nodes, actions
            nodes.append(node)
            action = self.tree_policy(node, self.c)
            if action is None:  # We are in an end-state. Terminate
                break
            # exploit node's pre-existing values:
            node = node.children[node.child_actions.index(action)] if action in node.child_actions else None
            state = node.state if node is not None else self.state_manager.do_action(state, action)
            actions.append(action)
            if node is None:
                node = self.node_search(self.root, state)  # search for state (many paths to state may exist)
                if node is not None:  # search succesful, join existing parent and child
                    self.insert_node(nodes[-1], node, self.state_manager.find_action(nodes[-1].state, state))
        return nodes, actions

    def new_node(self, state, level, end_state: bool = False): # create new node
        node_actions = [] if end_state else self.state_manager.get_actions(state, known_not_endstate=True)
        confidence = 0 if end_state else self.min_confidence
        node = Node(state, node_actions, confidence, level)
        return node

    def insert_node(self, parent_node: Node, child_node: Node, child_action: int): # combine parent and child nodes
        parent_node.children.append(child_node)
        parent_node.child_actions.append(child_action)

    def rollout_sim(self, state: Any): # rollout from leaf state
        actions = []
        while not self.state_manager.in_end_state(state):
            a = self.default_policy(state)
            actions.append(a)
            state = self.state_manager.do_action(state, a)
        return actions, self.state_manager.p1_reward(state, True)  # return actions and reward

    def backup(self, nodes: List[Node], actions: List[int], z):
        if len(nodes) > len(actions):  # if nodes contains endstate:
            nodes.pop()  # remove endstate
            self.Q[str(nodes[-1].state)][actions[-1]] = z  # adjust Q and amafQ to known ground truth
            self.amaf_Q[str(nodes[-1].state)][actions[-1]] = z
        for t, node in enumerate(nodes):
            node.N[actions[t]] += 1
            node.sum_N += 1
            self.Q[str(node.state)][actions[t]] += (z - self.Q[str(node.state)][actions[t]]) / node.N[actions[t]]
            for u in range(t + 2, len(actions), 2):
                node.amaf_N[actions[u]] += 1
                self.amaf_Q[str(node.state)][actions[u]] += (z - self.amaf_Q[str(node.state)][actions[u]]) / node.amaf_N[actions[u]]
            self.update_bias(str(node.state), actions[t], node.level)

    def evaluate(self, node, action, c):  # mc-rave eval with uct
        beta = node.amaf_N[action] / (node.N[action] + node.amaf_N[action] + 4 * node.N[action] * node.amaf_N[action] * self.biases[node.level][0] ** 2)
        return (1 - beta) * self.Q[str(node.state)][action] + beta * self.amaf_Q[str(node.state)][action] + c * sqrt(log(node.sum_N) / node.N[action])

    def tree_policy(self, node, c):  # return action with best eval
        if len(node.legal_actions) == 0:
            return None
        if self.state_manager.p1_to_move(node.state):
            return max(node.legal_actions, key=lambda a: self.evaluate(node, a, c))
        else:
            return min(node.legal_actions, key=lambda a: self.evaluate(node, a, -c))

    def root_distribution(self):
        return {action: self.root.N[action] / self.root.sum_N for action in self.root.legal_actions}  # return dict with root visit distribution

    def default_policy(self, state: Any) -> int:  # 'reasonably explorative'
        mask = self.state_manager.action_vector_mask(state)
        vector = self.state_manager.to_array(state)
        net_out = self.anet.forward(vector)[0]
        masked_out = np.multiply(net_out, mask) ** 2
        masked_out = np.divide(masked_out, np.sum(masked_out))
        return np.random.choice(np.arange(len(masked_out)), p=masked_out)  # use prob_dist to select default policy action

    def best_policy_action(self, node):  # return action with max Q value
        if self.state_manager.p1_to_move(node.state):
            return max(node.legal_actions, key=lambda a: self.Q[str(node.state)][a])
        else:
            return min(node.legal_actions, key=lambda a: self.Q[str(node.state)][a])

    def most_visited_child_action(self, node):  # return action with max visit count
        return max(node.child_actions, key=lambda a: node.N[a])

