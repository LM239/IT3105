from typing import List


class HexWorld:

    def __init__(self, hex_cfg):
        self.size = hex_cfg["size"]

        self.adjancencies = {}
        for y in range(self.size):
            for x in range(self.size):
                index = str(self.from2D(y, x))
                self.adjancencies[index] = []
                if self.valid_coords(y + 1, x):  # see if coordinate is within the board
                    self.adjancencies[index].append(self.from2D(y + 1, x))
                if self.valid_coords(y - 1, x):
                    self.adjancencies[index].append(self.from2D(y - 1, x))
                if self.valid_coords(y, x + 1):
                    self.adjancencies[index].append(self.from2D(y, x + 1))
                if self.valid_coords(y, x - 1):
                    self.adjancencies[index].append(self.from2D(y, x - 1))
                if self.valid_coords(y - 1, x + 1):
                    self.adjancencies[index].append(self.from2D(y - 1, x + 1))
                if self.valid_coords(y + 1, x - 1):
                    self.adjancencies[index].append(self.from2D(y + 1, x - 1))

    def valid_coords(self, y: int, x: int) -> bool:  # verifies that coorinates are within the board
        return 0 <= x < self.size and 0 <= y < self.size

    def new_world(self):
        return [(1, 0)].extend([(0, 0) for i in self.size ** 2])

    def get_actions(self, state):
        return [i for i, t in enumerate(state[1:]) if t[0] == t[1]]

    def do_action(self, state, action):
        return [reversed(state[0])] + state[1:action] + [state[0]] + state[action + 1:]

    def from2D(self, y: int, x: int) -> int:  # find 1D list index for a given 2D coordinate
        return (y * self.size) + x

    def from1D(self, index):
        return divmod(index, self.size) #TODO check validity

    def in_end_state(self, state: List) -> bool:
        return any(map(self.in_end_state_rec, list((state, i, "y", list(range(self.size))) for i in range(self.size) if state[i][0] == 1))) if state[0][0] == 1 \
               else any(map(self.in_end_state_rec, list((state, self.size * i, "x", list(range(self.size, self.size ** 2, self.size))) for i in range(self.size) if state[self.size * i][1] == 1)))

    def in_end_state_rec(self, data) -> bool:
        state, index, axis, path, = data
        if axis == "y":
            if index >= self.size * (self.size - 1):
                return True
            return any(map(self.in_end_state_rec, list((state, i, "y", path.append(i)) for i in self.adjancencies[str(index)] if state[i][0] == 1)))
        if index % self.size == self.size - 1:
            return True
        return any(map(self.in_end_state_rec, list((state, i, "x", path.append(i)) for i in self.adjancencies[str(index)] if state[i][1] == 1)))


