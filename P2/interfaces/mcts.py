from typing import Any

# Interface for mcts search
class Mcts:

    def root_distribution(self):
        pass

    def run_root(self, state: Any, use_og_root=False):
        pass

    def run_subtree(self, state: Any):
        pass

    def default_policy(self, state: Any) -> int:
        pass