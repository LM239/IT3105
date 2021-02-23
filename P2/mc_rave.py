import time
from typing import List, Tuple
from collections import defaultdict
from worlds.world import SimWorld
from search.treesearch import default_search

class Node:
    def __init__(self, state: List[Tuple[int, int]], heuristic):
        self.state: List[Tuple[int, int]] = state
        self.children: List[Node] = []
        self.N = defaultdict(heuristic)
        self.amaf_N = defaultdict(heuristic)
        self.child_actions: List[int] = []

class McRave:

    def __init__(self, mcts_cfg, state_manager, node_heuristic=lambda state: 10, node_search=default_search):
        self.bias: float = mcts_cfg["bias"]
        self.Q = defaultdict(lambda: 0.5)
        self.amaf_Q = {}
        self.search_duration = mcts_cfg["search_duration"]
        self.root: Node = None
        self.state_manager: SimWorld = state_manager
        self.node_heuristic = node_heuristic
        self.node_search = node_search

    def run_root(self, state: List[Tuple[int, int]]):
        now = time.time()
        self.root = Node(state, self.node_heuristic)
        while time.time() - now < self.search_duration:
            self.simulate(self.root)

    def run_subtree(self, state: List[Tuple[int, int]]):
        now = time.time()
        self.root = self.node_search(self.root, state)
        while time.time() - now < self.search_duration:
            self.simulate(self.root)

    def simulate(self, state):
        states, actions = self.sim_tree(state)
        rollout_actions, z = self.sim_default(states[-1], len(actions))
        self.backup(states, actions + rollout_actions, z)

    def sim_tree(self, state):
        t = 0
        actions = []
        node: Node = self.node_search(self.root, state)
        nodes = []
        while not self.state_manager.in_end_state(state):
            if node == None:
                node = Node(state, self.node_heuristic)
                action = self.default_policy(node)
                if (len(actions) > 0):
                    nodes[-1].child_actions.append(action)
                    nodes[-1].children.append(node)
                nodes.append(node)
                actions.append(action)
                return nodes, actions
            nodes.append(node)
            action = self.select_move(node)
            node = node.children[node.child_actions.index(action)] if action in node.child_actions else None
            state = self.state_manager.do_action(state, action)
            actions.append(action)
        return nodes, actions

    def sim_default(self, state: Node, t: int):
        return None, None

    def backup(self, states, actions, z):
        pass

    def default_policy(self, state):
        pass


"""
dfs(root, target):
    visited[]
    for a in root:
        if a not in target:
            return false
    stack = root
        dfs:
        node = stack.pop()
        goal_test: if node.num_moves = target.nummoves:
            return True
        for child in node.children:
            a = child_action
            if target[a] == child[a]:
                stack.append(c)
    return false

"""













