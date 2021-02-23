from interfaces.Node import Node
from collections import defaultdict

def default_search(root: Node, target):
    stack = [root]
    visited = defaultdict(lambda: False)
    while len(stack) > 0:
        v = stack.pop()
        if not visited[v]:
            if v.state == target:
                return v
            visited[v] = True
            for child in v.children:
                stack.append(child)
    return None


def hex_search(root: Node, target):
    for r1, t1 in zip(root.state[:-1], target[:-1]):
        if r1 != t1:
            return None
    target_moves = len([t for t in target[:-1] if t[0] != t[1]])
    stack = [root]
    visited = defaultdict(lambda: False)
    while len(stack) > 0:
        v = stack.pop()
        if not visited[v]:
            if len([t for t in v.state[:-1] if t[0] != t[1]]) == target_moves:
                return v
            visited[v] = True
            for action, child in zip(v.child_actions, v.children):
                if target[action] == child[action]:
                    stack.append(child)
    return None


