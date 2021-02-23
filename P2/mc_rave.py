import time
import random
from typing import List, Tuple, Any
from collections import defaultdict
from worlds.world import SimWorld
from search.treesearch import default_search
from Node import Node
from mcts import Mcts


class McRave(Mcts):

    def __init__(self, mcts_cfg, state_manager, node_heuristic=lambda: 10, node_search=default_search):
        self.bias: float = mcts_cfg["bias"]
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.5))
        self.amaf_Q = defaultdict(lambda: defaultdict(lambda: 0.5))
        self.search_duration = mcts_cfg["search_duration"]
        self.root: Node | None = None
        self.state_manager: SimWorld = state_manager
        self.node_heuristic = node_heuristic
        self.node_search = node_search
        self.epsilon = mcts_cfg["epsilon"]
        self.epsilon_decay = mcts_cfg["epsilon_decay"]
        self.epsilon_min = mcts_cfg["epsilon_min"]

    def run_root(self, state: Any):
        now = time.time()
        self.root = Node(state, self.node_heuristic)
        while time.time() - now < self.search_duration:
            self.simulate(self.root.state)
        self.epsilon *= self.epsilon_decay

    def run_subtree(self, state: Any):
        for child in self.root.children:
            if child.state == state:
                self.root = child
                break
        now = time.time()
        while time.time() - now < self.search_duration:
            self.simulate(self.root)

    def simulate(self, state):
        nodes, actions = self.tree_search(state)
        rollout_actions, z = self.rollout_sim(nodes[-1].state, len(actions))
        self.backup(nodes, actions + rollout_actions, z)

    def tree_search(self, state):
        actions = []
        node: Node = self.node_search(self.root, state)
        nodes = []
        while not self.state_manager.in_end_state(state):
            if node is None:
                node = Node(state, self.node_heuristic)
                action = self.default_policy(node.state)
                if len(actions) > 0:
                    nodes[-1].child_actions.append(action)
                    nodes[-1].children.append(node)
                nodes.append(node)
                actions.append(action)
                return nodes, actions
            nodes.append(node)
            action = self.tree_policy(node)
            node = node.children[node.child_actions.index(action)] if action in node.child_actions else None
            state = self.state_manager.do_action(state, action)
            actions.append(action)
            if node is None:
                node = self.node_search(self.root, state)
                if node is not None:
                    child_action = self.state_manager.find_action(nodes[-1].state, state)
                    nodes[-1].children.append(node)
                    nodes[-1].child_actions.append(child_action)
        return nodes, actions

    def rollout_sim(self, state: Any, t: int):
        actions = []
        while not self.state_manager.in_end_state(state):
            a = self.default_policy(state)
            actions.append(a)
            state = self.state_manager.do_action(state, a)
        return actions, self.state_manager.p1_reward(state)

    def backup(self, nodes: List[Node], actions: List[int], z):
        for t, node in enumerate(nodes):
            node.N[actions[t]] += 1
            self.Q[str(node.state)][actions[t]] += (z - self.Q[str(node.state)][actions[t]]) / (node.N[actions[t]])
            for u in range(t + 2, len(actions), 2):
                node.amaf_N[actions[u]] += 1
                self.amaf_Q[str(node.state)][actions[u]] += (z - self.amaf_Q[str(node.state)][actions[t]]) / (node.amaf_N[actions[t]])

    def evaluate(self, node, action):
        beta = node.amaf_N[action] / (node.N[action] + node.amaf_N[action] + 4 * node.N[action] * node.amaf_N[action] * self.bias ** 2)
        return (1 - beta) * self.Q[str(node.state)][action] + beta * self.amaf_Q[str(node.state)][action]

    def tree_policy(self, node):  # 'highly explorative'
        legal = self.state_manager.get_actions(node.state)
        best_a = None
        p = random.random()
        if p < self.epsilon + self.epsilon_min:
            return legal[random.randint(0, len(legal) - 1)]
        if self.state_manager.p1_to_move(node.state):
            best = float("-inf")
            for action in legal:
                score = self.evaluate(node, action)
                if score > best:
                    best = score
                    best_a = action
        else:
            best = float("inf")
            for action in legal:
                score = self.evaluate(node, action)
                if score < best:
                    best = score
                    best_a = action
        return best_a

    def default_policy(self, state: Any) -> int:  # 'reasonably explorative'
        legal = self.state_manager.get_actions(state)
        return legal[random.randint(0, len(legal)-1)]
