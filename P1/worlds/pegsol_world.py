import networkx as nx
import pylab as plt
from typing import List, Tuple
from configs.validate_configs import validate_pegsol_config


class PegSolitaire:

    def __init__(self, config):
        validate_pegsol_config(config)  # make sure config is valid

        self.type = config["type"]  # bord type "diamond" or triangle
        self.size = config["size"]  # board size size 3 or more

        if self.type == "triangle":
            self.state = [[1] * i for i in range(1, self.size + 1)]  # list representing triangle state - accessed with (y, x)
        else:
            self.state = [[1 for i in range(self.size)] for j in range(self.size)]  # n*n list representing diamond state

        self.display_rate = config["display_rate"] if "display_rate" in config else 0.5  # set display rate of episode playback
        self.adjacencies = []  # adjancency table pre-computes neighbours of all pegs
        for y in range(self.size):
            x_range = y + 1 if self.type == "triangle" else self.size  # find size of row (for triangle x_max for row y equals y)
            row = []
            self.adjacencies.append(row)
            for x in range(x_range): # check all possible neighbour pegs
                col = []
                row.append(col)
                if self.valid_coords(y + 1, x): # see if coordinate is within the board
                    col.append((y + 1, x))
                if self.valid_coords(y - 1, x):
                    col.append((y - 1, x))
                if self.valid_coords(y, x + 1):
                    col.append((y, x + 1))
                if self.valid_coords(y, x - 1):
                    col.append((y, x - 1))
                if self.type == "triangle":  # diagonal neighbours are different for triangle vs diamond board
                    if self.valid_coords(y + 1, x + 1):
                        col.append((y + 1, x + 1))
                    if self.valid_coords(y - 1, x - 1):
                        col.append((y - 1, x - 1))
                else:  # find diagonal neighbours for diamond board
                    if self.valid_coords(y - 1, x + 1):
                        col.append((y - 1, x + 1))
                    if self.valid_coords(y + 1, x - 1):
                        col.append((y + 1, x - 1))

        if "open_cells" in config:  # set open cells when given
            self.initial_open_cells = config["open_cells"]  # save open_cells to remember initial board state
            for cell in config["open_cells"]:
                if not (len(cell) == 2 and self.valid_coords(cell[0], cell[1])):  # validat given coordinates
                    print("Open cells must be specified as 2D coordinates (y,x) within the boards' bounds")
                    print("Erroneous coordinate: {}\nExiting".format(cell))
                    exit(0)
                self.state[cell[0]][cell[1]] = 0  # open the given cell
        elif self.size > 4 or (self.type == "diamond" and self.size > 3):  # when no open cell specified open one cell in the center
            row = int(self.size / 2)
            col = row if self.type == "diamond" else int(row / 2)
            self.state[row][col - 1] = 0
            self.initial_open_cells = [[row, col - 1]]
        else:  # if the board is too small a center hole is unsolvable, thus we open (0, 0)
            self.state[0][0] = 0
            self.initial_open_cells = [[0, 0]]
        self.episode = [self.vector()]  #  keep track of board state for one game
        self.peg_count = []  # keep track of peg count for all games

    def get_actions(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:  # return actions as a list of (from(y, x), to(y, x)) tuples
        actions = [] # list of available actions
        for y, row in enumerate(self.state):
            for x, peg in enumerate(row):
                if peg == 0:  # if the cell has no peg look elsewhere
                    continue
                for n_peg in self.adjacencies[y][x]:
                    if self.state[n_peg[0]][n_peg[1]] == 0:
                        continue # if the neighbour of interest has no peg, look to other neighboburs
                    to_y = 2 * n_peg[0] - y  #  doubles the diffeence between y and neigbours y to find to_y
                    to_x = 2 * n_peg[1] - x
                    if self.valid_coords(to_y, to_x) and self.state[to_y][to_x] == 0: # if the coordinates are valid and the peg is open add action
                        actions.append(((y, x), (to_y, to_x)))
        return actions

    def do_action(self, action: Tuple[Tuple[int, int], Tuple[int, int]]) -> None:  # perform action (does not validate tuple)
        to_pos = action[1]
        from_pos = action[0]

        self.state[from_pos[0]][from_pos[1]] = 0  # set from pos to 0
        self.state[to_pos[0]][to_pos[1]] = 1  # set to pos to 1

        #  average of to_pos and from_pos is set to o
        self.state[int((from_pos[0] + to_pos[0]) / 2)][int((from_pos[1] + to_pos[1]) / 2)] = 0
        self.episode.append(self.vector())  # store the new state in the game's state history

    def valid_coords(self, y: int, x: int) -> bool:  #  verifies that coorinates are within the board
        return 0 <= x < self.size and 0 <= y < self.size and ((not self.type == "triangle") or x <= y)

    def is_end_state(self) -> bool:  # return True when state iss endtsate, i-e no avaiable actions
        return len(self.get_actions()) == 0

    def state_reward(self) -> int:
        if not self.is_end_state():
            return 0  # no reward when no end state
        elif sum([peg for row in self.state for peg in row]) > 1:
            return -3  # punishment when end state is unfavourable
        return 5  # reward when en state is winning

    def __str__(self) -> str:
        return "".join(str(peg) for row in self.state for peg in row)  # flatten board state and return as bitstring

    def vector(self) -> List[int]:
        return [peg for row in self.state for peg in row]  # flatten board state and return as list / vector

    def reset(self):
        self.peg_count.append(sum(self.vector()))  # keep track of peg count for all gamess
        if self.type == "triangle":
            self.state = [[1] * i for i in range(1, self.size + 1)]  # reset board state
        else:
            self.state = [[1 for i in range(self.size)] for j in range(self.size)]
        for cell in self.initial_open_cells:
            self.state[cell[0]][cell[1]] = 0  # set open cells
        self.episode = [self.vector()]  # new list of sstate for the current game
        return self

    def visualize_peg_count(self):  # plot peg count for all games so far
        plt.plot(self.peg_count)
        plt.xlabel("Episode")
        plt.ylabel("Remaining pegs")
        plt.show()

    def visualize(self, states):  # visalize states (list of states)
        G = nx.Graph()
        for y in range(self.size):
            x_range = y + 1 if self.type == "triangle" else self.size
            for x in range(x_range):
                pos = (x - 0.5 * y, - y) if self.type == "triangle" else (x - 0.5 * (x + y), -x - y)  # use manhattan distance to find node positions
                G.add_node((y, x), pos=pos)
                for node in self.adjacencies[y][x]:  # use adjancency matrix to add edges
                    G.add_edge((y, x), node)
        pos = nx.get_node_attributes(G, 'pos')  # extract node positions
        size = (self.size, int(1.5 * self.size)) if self.type == "diamond" else (self.size, 0.7 * self.size)  # rough estimate of reasonable figure sixe
        plt.figure(figsize=size)  # set fig size
        for state in states:  # go through each state and visualize
            open_nodes = []  # set of nodes to color as open
            closed_nodes = []  # set of nodes to color as closed
            for y in range(self.size):
                x_range = y + 1 if self.type == "triangle" else self.size
                for x in range(x_range):
                    if state[self.from2D(y, x)] == 1:  #  add node to appropriate set
                        closed_nodes.append((y, x))
                    else:
                        open_nodes.append((y, x))
            nx.draw_networkx_nodes(G, pos, nodelist=open_nodes, node_color="g")  # draw open nodes (green)
            nx.draw_networkx_nodes(G, pos, nodelist=closed_nodes, node_color="r")  # draw closed nodes (red)
            nx.draw_networkx_labels(G, pos, font_weight="bold")  #  draw node names (their coordinate)
            nx.draw_networkx_edges(G, pos)  # draw edges
            plt.draw()  # dinish figure
            plt.pause(self.display_rate)  # delay before continuing to next state in states
            plt.clf()  # clear canvas
        plt.close() # close window

    def visualize_self(self) -> None:
        return self.visualize([self.vector()])  # visualize current state

    def visualize_episode(self) -> None:  # visualize episode so far
        return self.visualize(self.episode)

    def from2D(self, y: int, x: int) -> int:  # find 1D list index for a given 2D coordinate
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
    print(tri_world.get_actions())
    print(dim_world.get_actions())

    tri_world.visualize_self()
    # dim_world.do_action(((1, 3), (3, 3)))
    dim_world.visualize_self()
