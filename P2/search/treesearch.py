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
    root_moves = 0
    target_moves = 0
    for r1, t1 in zip(root.state[:-1], target[:-1]):
        if r1[0] != r1[1]:
            if r1 != t1:
                return None
            root_moves += 1
        if t1[0] != t1[1]:
            target_moves += 1
    stack = [root]
    all_v_moves = [root_moves]
    visited = defaultdict(lambda: False)
    while len(stack) > 0:
        v = stack.pop()
        v_moves = all_v_moves.pop()
        if not visited[str(v.state)]:
            if v_moves == target_moves:
                return v
            visited[v] = True
            for action, child in zip(v.child_actions, v.children):
                if target[action] == child.state[action]:
                    stack.append(child)
                    all_v_moves.append(v_moves + 1)
    return None


