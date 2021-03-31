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
        self.bias: float = mcts_cfg["bias"]
        self.Q = defaultdict(dd)
        self.amaf_Q = defaultdict(dd)

        try:
            self.amaf_Q = pickle.load(open("data/newrand_eps175/q_dicts/amaf_q.p", "rb"))
            self.Q = pickle.load(open("data/newrand_eps175/q_dicts/q.p", "rb"))
            print("Loaded amaf_q and Q dict with {} keys, and {} keys, respectively ({} bytes (not accurate))".format(len(self.amaf_Q), len(self.Q), sys.getsizeof(self.amaf_Q) + sys.getsizeof(self.Q)))
        except FileNotFoundError:
            print("Could not load q_dicts")
            pass

        self.search_games = mcts_cfg["search_games"]
        self.min_confidence = mcts_cfg["h_confidence"]
        self.search_duration = mcts_cfg["search_duration"]
        self.max_rollouts = mcts_cfg["max_rollouts"]
        self.root: Node | None = None
        self.og_root: Node | None = None
        self.state_manager: SimWorld = state_manager
        self.node_search = node_search
        self.c = mcts_cfg["c"]
        self.anet: ActorNet = anet


    dict1 = defaultdict(dd)

    def save_Qs(self, dir):
        os.makedirs(os.path.dirname(dir + "q_dicts/amaf_q.p"), exist_ok=True)
        with open(dir + "q_dicts/amaf_q.p", "wb") as file:
            pickle.dump(self.amaf_Q, file)
        os.makedirs(os.path.dirname(dir + "q_dicts/q.p"), exist_ok=True)
        with open(dir + "q_dicts/q.p", "wb") as file:
            pickle.dump(self.Q, file)

    def run_root(self, state: Any, use_og_root=False):
        if len(self.amaf_Q.keys()) + len(self.Q.keys()) > 4.5 * 10**6:
            print("\nPruning dicts")
            new_amaf_dict = defaultdict(dd)
            for key in self.amaf_Q.keys():
                if key.count("(0, 0)") >= 28:
                    new_amaf_dict[key] = self.amaf_Q[key]
            del self.amaf_Q
            self.amaf_Q = new_amaf_dict
            new_q_dict = defaultdict(dd)
            for key in self.Q.keys():
                if key.count("(0, 0)") >= 28:
                    new_q_dict[key] = self.Q[key]
            del self.Q
            self.Q = new_q_dict
            print("New amaf_q and Q dicts have {} keys, and {} keys, respectively ({} bytes (not accurate))".format(len(self.amaf_Q), len(self.Q), sys.getsizeof(self.amaf_Q) + sys.getsizeof(self.Q)))
        if use_og_root:
            self.root = self.og_root
        else:
            count = 0
            sum_of_diff = 0
            for key in self.Q.keys():
                count += len(self.Q[key].keys())
                for key2 in self.Q[key].keys():
                    sum_of_diff += abs(self.amaf_Q[key][key2] - self.Q[key][key2])
            if count > 0:
                print("\nActual, average bias: ", sum_of_diff / count)
                self.bias = sum_of_diff / count + 0.001
            print("\ngc freed {} objects".format(self.delete_tree()))
            self.root = self.new_node(state)
            self.og_root = self.root
        return self.simulate(self.root.state, self.search_duration, self.search_games)

    def run_subtree(self, state: Any):
        action = self.state_manager.find_action(self.root.state, state)
        if action in self.root.child_actions:
            self.root = self.root.children[self.root.child_actions.index(action)]
        else:
            self.root = self.new_node(state)
        return self.simulate(self.root.state, self.search_duration, self.search_games)

    def simulate(self, state, search_duration, num_searches, extended=False):
        rollouts = 0
        now = time.time()
        while True:
            nodes, actions = self.tree_search(state)
            rollout_actions, z = self.rollout_sim(nodes[-1].state)
            self.backup(nodes, actions + rollout_actions, z)
            rollouts += 1
            if time.time() - now > search_duration:
                if rollouts >= num_searches:
                    break
                now = time.time() - 3 * search_duration / 4
            if rollouts >= self.max_rollouts:
                break
        if not (extended or self.best_policy_action(self.root) == self.most_visited_child_action(self.root)):
            return rollouts + self.simulate(state, search_duration / 2, int(num_searches / 2), True)[0], 1
        return rollouts, 0

    def tree_search(self, state):
        node: Node = self.root
        actions: List[int] = []
        nodes: List[Node] = []
        while True:  # Stops if game is over
            if node is None:
                node = self.new_node(state, self.state_manager.in_end_state(state))
                self.insert_node(nodes[-1], node, actions[-1])
                nodes.append(node)
                return nodes, actions
            nodes.append(node)
            action = self.tree_policy(node, self.c)
            if action is None:  # We are in an end-state. Terminating
                break
            node = node.children[node.child_actions.index(action)] if action in node.child_actions else None
            state = node.state if node is not None else self.state_manager.do_action(state, action)
            actions.append(action)
            if node is None:
                node = self.node_search(self.root, state)
                if node is not None:
                    self.insert_node(nodes[-1], node, self.state_manager.find_action(nodes[-1].state, state))
        return nodes, actions

    def new_node(self, state, end_state: bool = False):
        node_actions = [] if end_state else self.state_manager.get_actions(state, known_not_endstate=True)
        confidence = 0 if end_state else self.min_confidence
        node = Node(state, node_actions, confidence)
        return node

    def insert_node(self, parent_node: Node, child_node: Node, child_action: int):
        parent_node.children.append(child_node)
        parent_node.child_actions.append(child_action)

    def rollout_sim(self, state: Any):
        actions = []
        while not self.state_manager.in_end_state(state):
            a = self.default_policy(state)
            actions.append(a)
            state = self.state_manager.do_action(state, a)
        return actions, self.state_manager.p1_reward(state, True)

    def backup(self, nodes: List[Node], actions: List[int], z):
        if len(nodes) > len(actions):
            nodes.pop()
            self.Q[str(nodes[-1].state)][actions[-1]] = z
            self.amaf_Q[str(nodes[-1].state)][actions[-1]] = z
        for t, node in enumerate(nodes):
            node.N[actions[t]] += 1
            node.sum_N += 1
            self.Q[str(node.state)][actions[t]] += (z - self.Q[str(node.state)][actions[t]]) / node.N[actions[t]]
            for u in range(t + 2, len(actions), 2):
                node.amaf_N[actions[u]] += 1
                self.amaf_Q[str(node.state)][actions[u]] += (z - self.amaf_Q[str(node.state)][actions[u]]) / node.amaf_N[actions[u]]

    def evaluate(self, node, action, c):
        beta = node.amaf_N[action] / (node.N[action] + node.amaf_N[action] + 4 * node.N[action] * node.amaf_N[action] * self.bias ** 2)
        return (1 - beta) * self.Q[str(node.state)][action] + beta * self.amaf_Q[str(node.state)][action] + c * sqrt(log(node.sum_N) / node.N[action])

    def tree_policy(self, node,  c):
        if len(node.legal_actions) == 0:
            return None
        if self.state_manager.p1_to_move(node.state):
            return max(node.legal_actions, key=lambda a: self.evaluate(node, a, c))
        else:
            return min(node.legal_actions, key=lambda a: self.evaluate(node, a, -c))

    def root_distribution(self):
        return {action: self.root.N[action] / self.root.sum_N for action in self.root.legal_actions}

    def default_policy(self, state: Any) -> int:  # 'reasonably explorative'
        mask = self.state_manager.action_vector_mask(state)
        vector = self.state_manager.to_array(state)
        net_out = self.anet.forward(vector)[0]
        masked_out = np.multiply(net_out, mask) ** 2
        masked_out = np.divide(masked_out, np.sum(masked_out))
        return np.random.choice(np.arange(len(masked_out)), p=masked_out)

    def best_policy_action(self, node):
        if self.state_manager.p1_to_move(node.state):
            return max(node.legal_actions, key=lambda a: self.Q[str(node.state)][a])
        else:
            return min(node.legal_actions, key=lambda a: self.Q[str(node.state)][a])

    def most_visited_child_action(self, node):
        return max(node.child_actions, key=lambda a: node.N[a])

    def delete_tree(self):
        stack: List[Node] = [self.root] if self.root is not None else []
        found = defaultdict(lambda: False)
        while len(stack) > 0:
            v = stack.pop(0)
            if not found[v]:
                found[v] = True
                stack.extend(v.children)
                del v
        del stack
        del found
        del self.root

        return gc.collect()
