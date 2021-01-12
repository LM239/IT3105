import networkx as nx
import pylab as plt
from networkx.drawing.nx_pydot import pydot_layout

class PegSolitaire:

    def __init__(self, config):
        if "type" not in config:
            print("Missing required PegSolitaire argument: 'type' \nExiting")
            exit(1)
        else:
            self.type = config["type"]
        if "size" not in config:
            print("Missing required PegSolitaire argument: 'size' \nExiting")
            exit(1)
        elif config["size"] < 3:
            print("Size parameter too small \nExiting")
            exit(1)
        else:
            self.size = config["size"]

        if self.type == "triangle":
            self.state = [[1] * i for i in range(1, self.size + 1)]
        elif self.type == "diamond":
            self.state = [[1 for i in range(self.size)] for j in range(self.size)]
        else:
            print("Unknown board type {}.\nExiting".format(self.type))
            exit(1)

        self.adjacencies = []
        for y in range(self.size):
            x_range = y + 1 if self.type == "triangle" else self.size
            row = []
            self.adjacencies.append(row)
            for x in range(x_range):
                col = []
                row.append(col)
                if self.valid_coors(y + 1, x):
                    col.append((y + 1, x))
                if self.valid_coors(y - 1, x):
                    col.append((y - 1, x))
                if self.valid_coors(y, x + 1):
                    col.append((y, x + 1))
                if self.valid_coors(y, x - 1):
                    col.append((y, x - 1))
                if self.type == "triangle":
                    if self.valid_coors(y + 1, x + 1):
                        col.append((y + 1, x + 1))
                    if self.valid_coors(y - 1, x - 1):
                        col.append((y - 1, x - 1))
                else:
                    if self.valid_coors(y - 1, x + 1):
                        col.append((y - 1, x + 1))
                    if self.valid_coors(y + 1, x - 1):
                        col.append((y + 1, x - 1))

        if "open_cells" in config:
            for cell in config["open_cells"]:
                if not len(cell) == 2:
                    print("Open cells must be specified as 2D coordinates (y,x)")
                    print("Erroneous coordinate: " + str(cell) + "\nExiting")
                    exit(0)
                try:
                    self.state[cell[0]][cell[1]] = 0
                except IndexError:
                    print("Cell at position ({}, {}) can not be open; it does not exist\nExiting".format(cell[0], cell[1]))
                    exit(1)
        self.episode = []

    def do_action(self, action):
        self.episode.append(str(self))
        return action

    def valid_coors(self, y, x):
        return 0 <= x < self.size and 0 <= y < self.size and ((not self.type == "triangle") or x <= y)

    def is_end_state(self):
        return self.is_end_state(self.state)

    def is_end_state(self, state):
        return state

    def __str__(self):
        return "".join(str(peg) for row in self.state for peg in row)

    def __int__(self):
        return int(str(self), 2)
    
    def vector(self):
        return [peg for row in self.state for peg in row]

    def visualize(self, states):
        G = nx.Graph()
        for y in range(self.size):
            x_range = y + 1 if self.type == "triangle" else self.size
            for x in range(x_range):
                G.add_node((y, x))
                for node in self.adjacencies[y][x]:
                    G.add_edge((y,x), node)
        for state in states:
            openNodes = []
            closedNodes = []
            state = list(state)
            for y in range(self.size):
                x_range = y + 1 if self.type == "triangle" else self.size
                for x in range(x_range):
                    if state[self.from2D(y,x)] == "1":
                        closedNodes.append((y, x))
                    else:
                        openNodes.append((y, x))

            pos = pydot_layout(G)
            nx.draw_networkx_nodes(G, pos, nodelist=openNodes, node_color="g")
            nx.draw_networkx_nodes(G, pos, nodelist=closedNodes, node_color="r")

            nx.draw_networkx_labels(G, pos, font_weight="bold")

            nx.draw_networkx_edges(G, pos)
            plt.show()

    def visualize_self(self):
        return self.visualize([str(self)])

    def visualize_episode(self):
        return self.visualize(self.episode)

    def from2D(self, y, x):
        if self.type == "triangle":
            return int((y * (y + 1) / 2) + x)
        else:
            return (y * self.size) + x


if __name__ == "__main__":
    tri_config = {
        "type": "triangle",
        "size": 4,
        "open_cells": [[0, 0], [3, 0], [3, 2], [2, 0], [2, 2]],
    }

    dim_config = {
        "type": "diamond",
        "size": 4,
        "open_cells": [[0, 0], [3, 0], [3, 2], [3, 3], [0, 3]],
    }

    tri_world = PegSolitaire(tri_config)
    dim_world = PegSolitaire(dim_config)

    print(tri_world.vector())
    print(tri_world)

    tri_world.visualize_self()
    dim_world.visualize_self()