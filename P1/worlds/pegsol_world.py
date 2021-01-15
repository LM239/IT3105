import networkx as nx
import pylab as plt
from typing import List, Tuple
from configs.validate_configs import validate_pegsol_config


class PegSolitaire:

    def __init__(self, config):
        validate_pegsol_config(config)

        self.type = config["type"]
        self.size = config["size"]

        if self.type == "triangle":
            self.state = [[1] * i for i in range(1, self.size + 1)]
        else:
            self.state = [[1 for i in range(self.size)] for j in range(self.size)]

        self.display_rate = config["display_rate"] if "display_rate" in config else 0.5
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
            self.initial_open_cells = config["open_cells"]
            for cell in config["open_cells"]:
                if not (len(cell) == 2 and self.valid_coords(cell[0], cell[1])):
                    print("Open cells must be specified as 2D coordinates (y,x) within the boards' bounds")
                    print("Erroneous coordinate: {}\nExiting".format(cell))
                    exit(0)
                self.state[cell[0]][cell[1]] = 0
        elif self.size > 4 or (self.type == "diamond" and self.size > 3):
            row = int(self.size / 2)
            col = row if self.type == "diamond" else int(row / 2)
            self.state[row][col - 1] = 0
            self.initial_open_cells = [[row, col - 1]]
        else:
            self.state[0][0] = 0
            self.initial_open_cells = [[0, 0]]
        self.episode = [self.vector()]
        self.peg_count = []

    def get_actions(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        actions = []
        for y, row in enumerate(self.state):
            for x, peg in enumerate(row):
                if peg == 0:
                    continue
                for n_peg in self.adjacencies[y][x]:
                    if self.state[n_peg[0]][n_peg[1]] == 0:
                        continue
                    to_y = 2 * n_peg[0] - y
                    to_x = 2 * n_peg[1] - x
                    if self.valid_coords(to_y, to_x) and self.state[to_y][to_x] == 0:
                        actions.append(((y, x), (to_y, to_x)))
        return actions

    def do_action(self, action: Tuple[Tuple[int, int], Tuple[int, int]]) -> None:
        to_pos = action[1]
        from_pos = action[0]

        self.state[from_pos[0]][from_pos[1]] = 0
        self.state[to_pos[0]][to_pos[1]] = 1

        self.state[int((from_pos[0] + to_pos[0]) / 2)][int((from_pos[1] + to_pos[1]) / 2)] = 0
        self.episode.append(self.vector())

    def valid_coords(self, y: int, x: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size and ((not self.type == "triangle") or x <= y)

    def is_end_state(self) -> bool:
        return len(self.get_actions()) == 0

    def state_reward(self) -> int:
        if not self.is_end_state():
            return 0
        elif sum([peg for row in self.state for peg in row]) > 1:
            return -100
        return 100

    def __str__(self) -> str:
        return "".join(str(peg) for row in self.state for peg in row)

    def __int__(self) -> int:
        return int(str(self), 2)

    def vector(self) -> List[int]:
        return [peg for row in self.state for peg in row]

    def reset(self):
        self.peg_count.append(sum(self.vector()))
        if self.type == "triangle":
            self.state = [[1] * i for i in range(1, self.size + 1)]
        else:
            self.state = [[1 for i in range(self.size)] for j in range(self.size)]
        for cell in self.initial_open_cells:
            self.state[cell[0]][cell[1]] = 0
        self.episode = [self.vector()]
        return self

    def visualize_peg_count(self):
        plt.plot(self.peg_count)
        plt.xlabel("Episode")
        plt.ylabel("Remaining pegs")
        plt.show()

    def visualize(self, states):
        G = nx.Graph()
        for y in range(self.size):
            x_range = y + 1 if self.type == "triangle" else self.size
            for x in range(x_range):
                pos = (x - 0.5 * y, - y) if self.type == "triangle" else (x - 0.5 * (x + y), -x - y)
                G.add_node((y, x), pos=pos)
                for node in self.adjacencies[y][x]:
                    G.add_edge((y, x), node)
        pos = nx.get_node_attributes(G, 'pos')
        size = (self.size, int(1.5 * self.size)) if self.type == "diamond" else (self.size, 0.7 * self.size)
        for state in states:
            open_nodes = []
            closed_nodes = []
            for y in range(self.size):
                x_range = y + 1 if self.type == "triangle" else self.size
                for x in range(x_range):
                    if state[self.from2D(y, x)] == 1:
                        closed_nodes.append((y, x))
                    else:
                        open_nodes.append((y, x))
            f, ax = plt.subplots(1, 1, figsize=size)
            nx.draw_networkx_nodes(G, pos, nodelist=open_nodes, node_color="g")
            nx.draw_networkx_nodes(G, pos, nodelist=closed_nodes, node_color="r")
            nx.draw_networkx_labels(G, pos, font_weight="bold")
            nx.draw_networkx_edges(G, pos)
            plt.draw()
            plt.pause(self.display_rate)

    def visualize_self(self) -> None:
        return self.visualize([self.vector()])

    def visualize_episode(self) -> None:
        return self.visualize(self.episode)

    def from2D(self, y: int, x: int) -> int:
        return int((y * (y + 1) / 2)) + x if self.type == "triangle" else (y * self.size) + x


if __name__ == "__main__":
    tri_config = {
        "type": "triangle",
        "size": 5,
        #"open_cells": [[0, 0], [3, 0], [3, 2]],
    }

    dim_config = {
        "type": "diamond",
        "size": 4,
        #"open_cells": [[0, 0], [3, 0], [3, 2], [3, 3], [0, 3]],
    }

    tri_world = PegSolitaire(tri_config)
    dim_world = PegSolitaire(dim_config)

    print(tri_world.vector())
    print(tri_world)
    print(tri_world.get_actions_self())
    print(dim_world.get_actions_self())

    tri_world.visualize_self()
    # dim_world.do_action(((1, 3), (3, 3)))
    dim_world.visualize_self()
