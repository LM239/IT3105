import time
import random
import numpy as np
from typing import List, Any
from collections import defaultdict
from interfaces.world import SimWorld
from search.treesearch import default_search
from interfaces.Node import Node
from interfaces.mcts import Mcts
from interfaces.actornet import ActorNet
from math import sqrt, log

class McRave(Mcts):

    def __init__(self, mcts_cfg, state_manager, anet, node_search=default_search):
        self.bias: float = mcts_cfg["bias"]
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.5))
        self.amaf_Q = defaultdict(lambda: defaultdict(lambda: 0.5))
        self.global_N = defaultdict(lambda: 0)
        self.max_confidence = mcts_cfg["max_h_confidence"]
        self.min_confidence = mcts_cfg["min_h_confidence"]
        self.amaf_confidence_scalar = mcts_cfg["amaf_conf_scalar"]
        self.search_duration = mcts_cfg["search_duration"]
        self.root: Node | None = None
        self.state_manager: SimWorld = state_manager
        self.node_search = node_search
        self.c = mcts_cfg["c"] #TODO eq (18) i paper, kan gjøre epsillon unødvendig
        self.anet: ActorNet = anet

    def run_root(self, state: Any):
        self.root = self.new_node(state)
        now = time.time()
        while time.time() - now < self.search_duration:
            self.simulate(self.root.state)

    def run_subtree(self, state: Any):
        action = self.state_manager.find_action(self.root.state, state)
        if action in self.root.child_actions:
            self.root = self.root.children[self.root.child_actions.index(action)]
        else:
            self.root = self.new_node(state)
        now = time.time()
        while time.time() - now < self.search_duration:
            self.simulate(self.root.state)
        print(self.root.sum_N)

    def simulate(self, state):
        nodes, actions = self.tree_search(state)
        rollout_actions, z = self.rollout_sim(nodes[-1].state)
        self.backup(nodes, actions + rollout_actions, z)

    def tree_search(self, state):
        node: Node = self.root
        actions: List[int] = []
        nodes: List[Node] = []
        while True: # Stops if game is over
            if node is None:
                node = self.new_node(state, self.state_manager.in_end_state(state))
                self.insert_node(nodes[-1], node, actions[-1])
                nodes.append(node)
                return nodes, actions
            nodes.append(node)
            action = self.tree_policy(node, self.c)
            if action is None: # We are in an end-state. Terminating
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
        node_actions = [] if end_state else self.state_manager.get_actions(state)
        confidence = 0 if end_state else self.min_confidence + self.global_N[str(state)] // len(node_actions)
        node = Node(state, node_actions, confidence, self.amaf_confidence_scalar * confidence)
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
            nodes = nodes[:-1]

        for t, node in enumerate(nodes):
            node.N[actions[t]] += 1
            node.sum_N += 1
            self.Q[node][actions[t]] += (z - self.Q[node][actions[t]]) / (node.N[actions[t]])
            if self.min_confidence + self.global_N[node] // len(node.legal_actions) < self.max_confidence:
                self.global_N[node] += 1
            for u in range(t + 2, len(actions), 2):
                node.amaf_N[actions[u]] += 1
                self.amaf_Q[node][actions[u]] += (z - self.amaf_Q[node][actions[u]]) / (node.amaf_N[actions[u]])

    def evaluate(self, node, action, c):
        beta = node.amaf_N[action] / (node.N[action] + node.amaf_N[action] + 4 * node.N[action] * node.amaf_N[action] * self.bias ** 2)
        return (1 - beta) * self.Q[node][action] + beta * self.amaf_Q[node][action] + c * sqrt(log(node.sum_N) / node.N[action])

    def tree_policy(self, node,  c):  # 'highly explorative'
        best_a = []
        if self.state_manager.p1_to_move(node.state):
            best = float("-inf")
            for action in node.legal_actions:
                score = self.evaluate(node, action, c)
                if score >= best:
                    if score > best:
                        best_a = []
                    best_a.append(action)
                    best = score
        else:
            best = float("inf")
            for action in node.legal_actions:
                score = self.evaluate(node, action, -c)
                if score <= best:
                    if score < best:
                        best_a = []
                    best_a.append(action)
                    best = score
        return best_a[random.randint(0, len(best_a)-1)] if len(best_a) > 0 else None

    def root_distribution(self):
        return {action: self.root.N[action] / self.root.sum_N for action in self.root.legal_actions}

    def default_policy(self, state: Any) -> int:  # 'reasonably explorative'
        mask = self.state_manager.action_vector_mask(state)
        vector = self.state_manager.to_array(state)
        net_out = self.anet.forward(vector)[0]
        masked_out = np.multiply(net_out, mask)
        masked_out = np.divide(masked_out, np.sum(masked_out))
        return np.random.choice(np.arange(len(masked_out)), p=masked_out)
        #         actions = self.state_manager.get_actions(state)
        #         return actions[random.randint(0, len(actions) - 1)]
