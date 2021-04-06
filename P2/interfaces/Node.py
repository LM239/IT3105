from collections import defaultdict
from typing import Any, List

# Interface for the nodes in the mcts search tree
class Node:
    def __init__(self, state: Any, actions: List[int], confidence: int, level: int):
        self.state: Any = state
        self.children: List[Node] = []
        self.N = defaultdict(lambda: confidence)
        self.amaf_N = defaultdict(lambda: confidence)
        self.child_actions: List[int] = []
        self.legal_actions: List[int] = actions
        self.sum_N: int = len(self.legal_actions) * confidence
        self.level = level