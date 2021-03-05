from collections import defaultdict
from typing import Any, List, Callable


class Node:
    def __init__(self, state: Any, actions: List[int], confidence: int, amaf_confidence: int):
        self.state: Any = state
        self.children: List[Node] = []
        self.N = defaultdict(lambda: confidence)
        self.amaf_N = defaultdict(lambda: amaf_confidence)
        self.child_actions: List[int] = []
        self.legal_actions: List[int] = actions
        self.sum_N: int = confidence * len(actions)
