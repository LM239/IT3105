from collections import defaultdict
from typing import Any, List, Callable


class Node:
    def __init__(self, state: Any, actions: List[int], heuristic: Callable[[], int]):
        self.state: Any = state
        self.children: List[Node] = []
        self.N = defaultdict(heuristic)
        self.amaf_N = defaultdict(heuristic)
        self.child_actions: List[int] = []
        self.legal_actions: List[int] = actions
        self.sum_N: int = heuristic() * len(actions)
