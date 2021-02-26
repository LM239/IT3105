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

    def __init__(self, mcts_cfg, state_manager, anet, node_heuristic=lambda: 3, node_search=default_search):
        self.bias: float = mcts_cfg["bias"]
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.5))
        self.amaf_Q = defaultdict(lambda: defaultdict(lambda: 0.5))
        self.search_duration = mcts_cfg["search_duration"]
        self.root: Node | None = None
        self.state_manager: SimWorld = state_manager
        self.node_heuristic = node_heuristic
        self.node_search = node_search
        self.c = mcts_cfg["c"] #TODO eq (18) i paper, kan gjøre epsillon unødvendig
        self.anet: ActorNet = anet

    def run_root(self, state: Any):
        now = time.time()
        self.root = Node(state, self.state_manager.get_actions(state), self.node_heuristic)
        while time.time() - now < self.search_duration:
            self.simulate(self.root.state)
        return self.tree_policy(self.root, 0)

    def run_subtree(self, state: Any):
        print(self.root.sum_N)
        for child in self.root.children:
            if child.state == state:
                self.root = child
                break
        else:
            self.root = Node(state, self.state_manager.get_actions(state), self.node_heuristic)
        now = time.time()
        while time.time() - now < self.search_duration:
            self.simulate(self.root.state)
        return self.tree_policy(self.root, 0)

    def simulate(self, state):
        nodes, actions = self.tree_search(state)
        rollout_actions, z = self.rollout_sim(nodes[-1].state)
        self.backup(nodes, actions + rollout_actions, z)

    def tree_search(self, state):
        node: Node = self.root
        actions: List[int] = []
        nodes: List[Node] = []
        while not self.state_manager.in_end_state(state):
            if node is None:
                node = Node(state, self.state_manager.get_actions(state), self.node_heuristic)
                self.insert_node(nodes[-1], node, actions[-1])
                action = self.default_policy(node.state)
                nodes.append(node)
                actions.append(action)
                return nodes, actions
            nodes.append(node)
            action = self.tree_policy(node, self.c)
            node = node.children[node.child_actions.index(action)] if action in node.child_actions else None
            state = self.state_manager.do_action(state, action)
            actions.append(action)
            if node is None:
                node = self.node_search(self.root, state)
                if node is not None:
                    self.insert_node(nodes[-1], node, self.state_manager.find_action(nodes[-1].state, state))
        return nodes, actions

    def insert_node(self, parent_node: Node, child_node: Node, child_action: int):
        parent_node.children.append(child_node)
        parent_node.child_actions.append(child_action)

    def rollout_sim(self, state: Any):
        actions = []
        while not self.state_manager.in_end_state(state):
            a = self.default_policy(state)
            actions.append(a)
            state = self.state_manager.do_action(state, a)
        return actions, self.state_manager.p1_reward(state)

    def backup(self, nodes: List[Node], actions: List[int], z):
        for t, node in enumerate(nodes):
            node.N[actions[t]] += 1
            node.sum_N += 1
            self.Q[str(node.state)][actions[t]] += (z - self.Q[str(node.state)][actions[t]]) / (node.N[actions[t]])
            for u in range(t + 2, len(actions), 2):
                node.amaf_N[actions[u]] += 1
                self.amaf_Q[str(node.state)][actions[u]] += (z - self.amaf_Q[str(node.state)][actions[t]]) / (node.amaf_N[actions[t]])

    def evaluate(self, node, action, c):
        beta = node.amaf_N[action] / (node.N[action] + node.amaf_N[action] + 4 * node.N[action] * node.amaf_N[action] * self.bias ** 2)
        return (1 - beta) * self.Q[str(node.state)][action] + beta * self.amaf_Q[str(node.state)][action] + c * sqrt(log(node.sum_N) / node.N[action])

    def tree_policy(self, node,  c):  # 'highly explorative'
        best_a = None
        if self.state_manager.p1_to_move(node.state):
            best = float("-inf")
            for action in node.legal_actions:
                score = self.evaluate(node, action, c)
                if score > best:
                    best = score
                    best_a = action
        else:
            best = float("inf")
            for action in node.legal_actions:
                score = self.evaluate(node, action, -c)
                if score < best:
                    best = score
                    best_a = action
        return best_a

    def root_distribution(self):
        dist = {}
        for action in self.state_manager.get_actions(self.root.state):
            dist[action] = self.root.N[action] / self.root.sum_N
        return dist

    def default_policy(self, state: Any) -> int:  # 'reasonably explorative'
        mask = self.state_manager.action_vector_mask(state)
        vector = self.state_manager.vector(state)
        net_out = self.anet.forward(vector)[0][0]
        masked_out = np.multiply(net_out, mask)
        masked_out = np.divide(masked_out, np.sum(masked_out))
        return np.random.choice(np.arange(len(masked_out)), p=masked_out)
        #         actions = self.state_manager.get_actions(state)
        #         return actions[random.randint(0, len(actions) - 1)]
