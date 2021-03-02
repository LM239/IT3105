from typing import List, Tuple
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib import patches
from interfaces.world import SimWorld
from collections import defaultdict
import random
import numpy as np

class HexWorld(SimWorld):

    def __init__(self, hex_cfg, display_rate):
        self.size = hex_cfg["size"]
        self.display_rate = display_rate

        self.adjacencies = {}
        self.adjacencies_xsort = {}
        self.adjacencies_ysort = {}
        for y in range(self.size):
            for x in range(self.size):
                index = str(self.from2D(y, x))
                self.adjacencies[index] = []
                if self.valid_coords(y + 1, x):  # see if coordinate is within the board
                    self.adjacencies[index].append(self.from2D(y + 1, x))
                    if self.valid_coords(y + 1, x - 1):
                        self.adjacencies[index].append(self.from2D(y + 1, x - 1))
                if self.valid_coords(y, x + 1):
                    self.adjacencies[index].append(self.from2D(y, x + 1))
                if self.valid_coords(y, x - 1):
                    self.adjacencies[index].append(self.from2D(y, x - 1))
                if self.valid_coords(y - 1, x):
                    if self.valid_coords(y - 1, x + 1):
                        self.adjacencies[index].append(self.from2D(y - 1, x + 1))
                    self.adjacencies[index].append(self.from2D(y - 1, x))
                self.adjacencies_ysort[index] = list(reversed(self.adjacencies[index]))
                self.adjacencies_xsort[index] = sorted(self.adjacencies[index], key=(lambda x: x % self.size))

    def valid_coords(self, y: int, x: int) -> bool:  # verifies that coordinates are within the board
        return 0 <= x < self.size and 0 <= y < self.size

    def new_state(self):
        return [(0, 0) for i in range(self.size ** 2)] + [(1, 0)]

    def get_actions(self, state):
        return [i for i, t in enumerate(state[:-1]) if t[0] == t[1]] if not self.in_end_state(state) else []

    def do_action(self, state, action):
        return state[0:action] + [state[-1]] + state[action + 1:-1] + [tuple(reversed(state[-1]))]

    def find_action(self, parent_state, child_state):
        for action, (square1, square2) in enumerate(zip(parent_state[:-1], child_state[:-1])):
            if square1 != square2:
                return action
        return None

    def from2D(self, y: int, x: int) -> int:  # find 1D list index for a given 2D coordinate
        return (y * self.size) + x

    def from1D(self, index):
        return divmod(index, self.size)

    def in_end_state(self, state):
        stack = [x for x in range(0, self.size ** 2, self.size) if state[x][1] == 1] if state[-1][0] == 1 else [x for x in range(self.size) if state[x][0] == 1]
        visits = defaultdict(lambda: False)
        goal_test = (lambda x: x % self.size == self.size - 1) if state[-1][0] == 1 else (lambda x: x >= self.size * (self.size - 1))
        c_index = 1 if state[-1][0] == 1 else 0
        e = self.adjacencies_xsort if state[-1][0] == 1 else self.adjacencies_ysort
        while len(stack) > 0:
            v = stack.pop()
            if not visits[v]:
                if goal_test(v):
                    return True
                visits[v] = True
                for c in e[str(v)]:
                    if state[c][c_index] == 1 and not visits[c]:
                        stack.append(c)
        return False

    def bfs(self, state):
        queue = [x for x in range(0, self.size ** 2, self.size) if state[x][1] == 1] if state[-1][0] == 1 else [x for x in range(self.size) if state[x][0] == 1]
        found = defaultdict(lambda: False)
        parents = defaultdict(lambda: -1)
        goal_test = (lambda x: x % self.size == self.size - 1) if state[-1][0] == 1 else (lambda x: x >= self.size * (self.size - 1))
        c_index = 1 if state[-1][0] == 1 else 0
        e = self.adjacencies_xsort if state[-1][0] == 1 else self.adjacencies_ysort
        for v in queue:
            found[v] = True
        while len(queue) > 0:
            v = queue.pop(0)
            if goal_test(v):
                path = [v]
                while parents[v] >= 0:
                    v = parents[v]
                    path.append(v)
                return path
            for c in e[str(v)]:
                if state[c][c_index] == 1 and not found[c]:
                    parents[c] = v
                    found[c] = True
                    queue.append(c)
        return []

    def winner(self, state):
        return tuple(reversed(state[-1])) if self.in_end_state(state) else None

    def p1_reward(self, state):
        return state[-1][1] if self.in_end_state(state) else None

    def p1_to_move(self, state):
        return state[-1][0] == 1

    def visualize(self, states, player_labels=("Player 1", "Player 2")):  # visualize states (list of states)
        G = nx.Graph()
        for i in range(self.size ** 2):
            y, x = self.from1D(i)
            pos = (x - 0.5 * (x + y), -x - y)  # use manhattan distance to find node positions
            G.add_node(i, pos=pos)
            for node in self.adjacencies[str(i)]:  # use adjacency matrix to add edges
                if node < i:
                    break
                if (i < self.size and node < self.size or i >= self.size * (self.size - 1) and node >= self.size * (self.size - 1)):
                    G.add_edge(i, node, width=2, color="green")
                elif (i % self.size == 0 and node % self.size == 0 or i % self.size == self.size - 1 and node % self.size == self.size - 1):
                    G.add_edge(i, node, width=2, color="red")
                else:
                    G.add_edge(i, node, width=1, color="black")
        pos = nx.get_node_attributes(G, 'pos')  # extract node positions
        red_patch = patches.Patch(color='red', label=player_labels[1])
        green_patch = patches.Patch(color='green', label=player_labels[0])
        plt.figure(figsize=(min(41, self.size), int(1.5 * min(41, self.size))))  # set fig size
        edge_widths = list(nx.get_edge_attributes(G, 'width').values())
        edge_colors = list(nx.get_edge_attributes(G, 'color').values())
        for i, state in enumerate(states):  # go through each state and visualize
            if i == len(states) - 1:
                path = self.bfs(state)
                if len(path) > 0:
                    for i in range(1, len(path)):
                        G.add_edge(path[i], path[i - 1], width=5, color="black")
                    edge_colors = list(nx.get_edge_attributes(G, 'color').values())
                    edge_widths = list(nx.get_edge_attributes(G, 'width').values())
            p1_nodes = []  # set of nodes to color as open
            p2_nodes = []
            empty_nodes = []  # set of nodes to color as closed
            for i in range(self.size ** 2):
                if state[i][0] == 1:  # add node to appropriate set
                    p1_nodes.append(i)
                elif state[i][1] == 1:
                    p2_nodes.append(i)
                else:
                    empty_nodes.append(i)

            nx.draw_networkx_nodes(G, pos, nodelist=p1_nodes, node_color="g", node_size=150)  # draw open nodes (green)
            nx.draw_networkx_nodes(G, pos, nodelist=p2_nodes, node_color="r", node_size=150)  # draw closed nodes (red)
            nx.draw_networkx_nodes(G, pos, nodelist=empty_nodes, node_color="y", node_size=150)  # draw closed nodes (red)
            #nx.draw_networkx_labels(G, pos, font_weight="bold")  # draw node names (their coordinate)
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)  # draw edges
            plt.legend(handles=[green_patch, red_patch], prop={'size': 2 * min(40, self.size) + 2})
            plt.draw()  # finish figure
            plt.pause(self.display_rate)  # delay before continuing to next state in states
            plt.clf()  # clear canvas
        #plt.close()  # close window

    def vector(self, state: List[Tuple[int, int]]) -> List[int]:
        return [val for tuple in state for val in tuple]  # flatten board state and return as list / vector

    def action_vector_mask(self, state):
        return [1 if state[i] == (0, 0) else 0 for i in range(self.size ** 2)]

    def complete_action_dist(self, action_dist):
        return [action_dist[i] if i in action_dist.keys() else 0 for i in range(self.size ** 2)]

    def to_array(self, state):
        array = np.zeroes(size=(self.size * self.size))
        for i in range(self.size):
            for j in range(self.size):
                t = state[self.from2D(i, j)]
                array[i][j] += t[0] - t[1]
        return array


if __name__ == "__main__":
    cfg = {
        "size": 3
    }
    game = HexWorld(cfg, 0.3)

    states = []
    state = game.new_state()
    actions = game.get_actions(state)
    print(game.find_action(state, game.do_action(state, 24)))
    while len(actions) > 0:
        state = game.do_action(state, actions[random.randint(0, len(actions) - 1)])
        actions = game.get_actions(state)
    states.append(state)

    print(game.vector(state))
    game.visualize([[(1, 0), (0, 0), (0, 0), (0, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0)]])




