from typing import List, Tuple
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib import patches
from world import SimWorld
from collections import defaultdict
import random


class HexWorld(SimWorld):

    def __init__(self, hex_cfg, display_rate):
        self.size = hex_cfg["size"]
        self.display_rate = display_rate
        self.paths = defaultdict(lambda: [])

        self.adjacencies = {}
        for y in range(self.size):
            for x in range(self.size):
                index = str(self.from2D(y, x))
                self.adjacencies[index] = []
                if self.valid_coords(y + 1, x):  # see if coordinate is within the board
                    self.adjacencies[index].append(self.from2D(y + 1, x))
                    if self.valid_coords(y + 1, x - 1):
                        self.adjacencies[index].append(self.from2D(y + 1, x - 1))
                if self.valid_coords(y - 1, x):
                    self.adjacencies[index].append(self.from2D(y - 1, x))
                    if self.valid_coords(y - 1, x + 1):
                        self.adjacencies[index].append(self.from2D(y - 1, x + 1))
                if self.valid_coords(y, x + 1):
                    self.adjacencies[index].append(self.from2D(y, x + 1))
                if self.valid_coords(y, x - 1):
                    self.adjacencies[index].append(self.from2D(y, x - 1))

    def valid_coords(self, y: int, x: int) -> bool:  # verifies that coordinates are within the board
        return 0 <= x < self.size and 0 <= y < self.size

    def new_state(self):
        return [(0, 0) for i in range(self.size ** 2)] + [(1, 0)]

    def get_actions(self, state):
        return [i for i, t in enumerate(state[:-1]) if t[0] == t[1]] if not self.in_end_state(state) else []

    def do_action(self, state, action):
        return state[0:action] + [state[-1]] + state[action + 1:-1] + [tuple(reversed(state[-1]))]

    def from2D(self, y: int, x: int) -> int:  # find 1D list index for a given 2D coordinate
        return (y * self.size) + x

    def from1D(self, index):
        return divmod(index, self.size)

    def in_end_state(self, state: List) -> bool:
        return any(map(self.in_end_state_rec_y, ((state, i, list(range(self.size)), i) for i in range(self.size) if state[i][0] == 1))) if state[-1][1] == 1 \
               else any(map(self.in_end_state_rec_x, ((state, self.size * i, list(range(0, self.size ** 2, self.size)), self.size * i) for i in range(self.size) if state[self.size * i][1] == 1)))

    def in_end_state_rec_y(self, data) -> bool:
        state, index, path, start = data
        if index >= self.size * (self.size - 1):
            self.paths[str(state)] = [start] + path[self.size:]
            return True
        return any(map(self.in_end_state_rec_y, ((state, i, path + [i], start) for i in self.adjacencies[str(index)] if state[i][0] == 1 and i not in path)))

    def in_end_state_rec_x(self, data) -> bool:
        state, index, path, start = data
        if index % self.size == self.size - 1:
            self.paths[str(state)] = [start] + path[self.size:]
            return True
        return any(map(self.in_end_state_rec_x, ((state, i, path + [i], start) for i in self.adjacencies[str(index)] if state[i][1] == 1 and i not in path)))

    def winner(self, state):
        return tuple(reversed(state[-1])) if self.in_end_state(state) else None

    def visualize(self, states):  # visualize states (list of states)
        G = nx.Graph()
        for i in range(self.size ** 2):
            y, x = self.from1D(i)
            pos = (x - 0.5 * (x + y), -x - y)  # use manhattan distance to find node positions
            G.add_node(i, pos=pos)
            for node in self.adjacencies[str(i)]:  # use adjacency matrix to add edges
                if node < i:
                    continue
                if (i < self.size and node < self.size or i >= self.size * (self.size - 1) and node >= self.size * (self.size - 1)):
                    G.add_edge(i, node, width=2, color="green")
                elif (i % self.size == 0 and node % self.size == 0 or i % self.size == self.size - 1 and node % self.size == self.size - 1):
                    G.add_edge(i, node, width=2, color="red")
                else:
                    G.add_edge(i, node, width=1, color="black")
        pos = nx.get_node_attributes(G, 'pos')  # extract node positions
        red_patch = patches.Patch(color='red', label='Player 2')
        green_patch = patches.Patch(color='green', label='Player 1')
        plt.figure(figsize=(self.size, int(1.5 * self.size)))  # set fig size
        edge_widths = list(nx.get_edge_attributes(G, 'width').values())
        edge_colors = list(nx.get_edge_attributes(G, 'color').values())
        for i, state in enumerate(states):  # go through each state and visualize
            if i == len(states) - 1:
                path = self.paths[str(state)]
                if len(path) > 0:
                    i = 2
                    while i < len(path):
                        if path[i - 2] in self.adjacencies[str(path[i])]:
                            path = path[:i - 1] + path[i:]
                        else:
                            i += 1
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
            nx.draw_networkx_nodes(G, pos, nodelist=p1_nodes, node_color="g")  # draw open nodes (green)
            nx.draw_networkx_nodes(G, pos, nodelist=p2_nodes, node_color="r")  # draw closed nodes (red)
            nx.draw_networkx_nodes(G, pos, nodelist=empty_nodes, node_color="y")  # draw closed nodes (red)
            nx.draw_networkx_labels(G, pos, font_weight="bold")  # draw node names (their coordinate)
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)  # draw edges
            plt.legend(handles=[green_patch, red_patch], prop={'size': 2 * self.size + 2})
            plt.draw()  # finish figure
            plt.pause(self.display_rate)  # delay before continuing to next state in states
            plt.clf()  # clear canvas
        plt.close()  # close window

    def vector(self, state: List[Tuple[int, int]]) -> List[int]:
        return [val for tuple in state for val in tuple]  # flatten board state and return as list / vector

    def action_vector(self, state):
        return [1 if state[i] == (0, 0) else 0 for i in range(self.size ** 2)]


if __name__ == "__main__":
    cfg = {
        "size": 5
    }
    game = HexWorld(cfg, 0.3)

    states = []
    state = game.new_state()
    actions = game.get_actions(state)
    print(game.child_states(state))
    while len(actions) > 0:
        states.append(state)
        print(actions)
        state = game.do_action(state, actions[random.randint(0, len(actions) - 1)])
        actions = game.get_actions(state)
    states.append(state)
    game.visualize([state])




