import networkx

class PegSolitaire:

    def __init__(self, config):
        if "type" not in config:
            print("Missing sim_world argument: 'type' \nExiting")
            exit(1)
        if "size" not in config:
            print("Missing sim_world argument: 'size' \nExiting")
            exit(1)
        elif config["size"] < 3:
            print("Size parameter too small \nExiting")
            exit(1)

        if config["type"] == "triangle":
            self.state = [[1] * i for i in range(1, config["size"] + 1)]

        elif config["type"] == "diamond":
            self.state = [[1] * config["size"]] * config["size"]
        else:
            print("Unknown board type {}.\nExiting".format(config["type"]))
            exit(1)

        if "open_cells" in config:
            for cell in config["open_cells"]:
                if not len(cell) == 2:
                    print("Open cells must be specified as 2D coordinates (y,x)")
                    print("Erroneous coordinate: " + str(cell) + "\nExiting")
                    exit(0)
                try:
                    self.state[cell[0]][cell[1]] = 0
                except IndexError:
                    print("Cell at position ({}, {}) can not be opened; it does not exist\nExiting".format(cell[0], cell[1]))
                    exit(1)

    def do_action(self, action):
        return action

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
        for state in states:
            self.visualize(state)

    def visualize(self):
        return 0


if __name__ == "__main__":
    tri_config = {
        "type": "triangle",
        "size": 4,
        "open_cells": [[0, 0], [3, 0]],
    }

    dim_config = {
        "type": "diamond",
        "size": 4,
    }

    tri_world = PegSolitaire(tri_config)
    dim_world = PegSolitaire(dim_config)

    print(tri_world.vector())
    print(tri_world)
    print(int(tri_world))