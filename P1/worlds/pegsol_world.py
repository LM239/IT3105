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
                if self.valid_coords(y + 1, x):
                    col.append((y + 1, x))
                if self.valid_coords(y - 1, x):
                    col.append((y - 1, x))
                if self.valid_coords(y, x + 1):
                    col.append((y, x + 1))
                if self.valid_coords(y, x - 1):
                    col.append((y, x - 1))
                if self.type == "triangle":
                    if self.valid_coords(y + 1, x + 1):
                        col.append((y + 1, x + 1))
                    if self.valid_coords(y - 1, x - 1):
                        col.append((y - 1, x - 1))
                else:
                    if self.valid_coords(y - 1, x + 1):
                        col.append((y - 1, x + 1))
                    if self.valid_coords(y + 1, x - 1):
                        col.append((y + 1, x - 1))

        if "open_cells" in config:
            for cell in config["open_cells"]:
                if not (len(cell) == 2 and self.valid_coords(cell[0], cell[1])):
                    print("Open cells must be specified as 2D coordinates (y,x) within the boards' bounds")
                    print("Erroneous coordinate: " + str(cell) + "\nExiting")
                    exit(0)
                self.state[cell[0]][cell[1]] = 0
        self.episode = [str(self.state)]

    def get_actions_self(self):
        return self.get_actions(self.state)

    def get_actions(self, state):
        actions = []
        for y, row in enumerate(state):
            for x, peg in enumerate(row):
                if peg == 0:
                    continue
                else:
                    for n_peg in self.adjacencies[y][x]:
                        if state[n_peg[0]][n_peg[1]] == 0:
                            continue
                        else:
                            action_y = - y + 2 * n_peg[0]
                            action_x = - x + 2 * n_peg[1]
                            if self.valid_coords(action_y, action_x) \
                                    and state[action_y][action_x] == 0:
                                actions.append(((y, x), (action_y, action_x)))
        return actions

    def do_action(self, action):
        to_pos = action[1]
        from_pos = action[0]

        self.state[from_pos[0]][from_pos[1]] = 0
        self.state[to_pos[0]][to_pos[1]] = 1

        dif_y = (to_pos[0] - from_pos[0]) / 2
        dif_x = (to_pos[1] - from_pos[1]) / 2

        self.state[from_pos[0] + int(dif_y)][from_pos[1] + int(dif_x)] = 0
        self.episode.append(str(self))
        return str(self)

    def valid_coords(self, y, x):
        return 0 <= x < self.size and 0 <= y < self.size and ((not self.type == "triangle") or x <= y)

    def is_end_state_self(self):
        return self.is_end_state(self.state)

    def is_end_state(self, state):
        return len(self.get_actions(state)) == 0

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
                    G.add_edge((y, x), node)
        for state in states:
            open_nodes = []
            closed_nodes = []
            state = list(state)
            for y in range(self.size):
                x_range = y + 1 if self.type == "triangle" else self.size
                for x in range(x_range):
                    if state[self.from2D(y, x)] == "1":
                        closed_nodes.append((y, x))
                    else:
                        open_nodes.append((y, x))

            pos = pydot_layout(G)
            nx.draw_networkx_nodes(G, pos, nodelist=open_nodes, node_color="g")
            nx.draw_networkx_nodes(G, pos, nodelist=closed_nodes, node_color="r")

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
        "open_cells": [[0, 0], [3, 0], [3, 2]],
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
    print(tri_world.get_actions_self())
    print(dim_world.get_actions_self())

    tri_world.visualize_self()
    dim_world.visualize_self()
    dim_world.do_action(((1, 3), (3, 3)))
    dim_world.visualize_self()
