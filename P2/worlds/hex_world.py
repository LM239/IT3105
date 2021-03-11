from typing import List, Tuple
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib import patches
from numpy import float32

from interfaces.world import AdvancedSimWorld
from collections import defaultdict
import random
import numpy as np

class HexWorld(AdvancedSimWorld):

    def __init__(self, hex_cfg, display_rate):
        self.size = hex_cfg["size"]
        self.display_rate = display_rate


        self.bridges = defaultdict(lambda: [])
        self.adjacencies = {}
        self.adjacencies_xsort = {}
        self.adjacencies_ysort = {}
        for y in range(self.size):
            for x in range(self.size):
                index = self.from2D(y, x)
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
                        if self.valid_bridge_coords(y - 1, x + 2): # known true: and self.valid_coords(y - 1, x + 1)
                            self.bridges[index].append(((y, x, index), (y, x + 1, self.from2D(y, x + 1)),
                                                        (y - 1, x + 1, self.from2D(y - 1, x + 1)),
                                                        (y - 1, x + 2, self.from2D(y - 1, x + 2))))
                        if self.valid_bridge_coords(y - 2, x + 1): # known true: and self.valid_coords(y - 1, x + 1):
                            self.bridges[index].append(((y, x, index), (y - 1, x, self.from2D(y - 1, x)),
                                                        (y - 1, x + 1, self.from2D(y - 1, x + 1)),
                                                        (y - 2, x + 1, self.from2D(y - 2, x + 1))))
                    self.adjacencies[index].append(self.from2D(y - 1, x)) #TODO'nt dont move anything!!!
                    if self.valid_bridge_coords(y - 1, x - 1) and self.valid_coords(y, x - 1): # known true: and self.valid_coords(y - 1, x)
                        self.bridges[index].append(((y, x, index), (y - 1, x, self.from2D(y - 1, x)),
                                                    (y, x - 1, self.from2D(y, x - 1)),
                                                    (y - 1, x - 1, self.from2D(y - 1, x - 1))))
                self.adjacencies_ysort[index] = list(reversed(self.adjacencies[index]))
                self.adjacencies_xsort[index] = sorted(self.adjacencies[index], key=(lambda x: x % self.size))
        y = self.size - 2
        index = self.from2D(y, 1)
        for x in range(1, self.size):
            self.bridges[index].append(((y, x, index), (y + 1, x, self.from2D(y + 1, x)),
                                        (y + 1, x - 1, self.from2D(y + 1, x - 1)),
                                        (y + 2, x - 1, self.from2D(y + 2, x - 1))))
            index += 1
        x = 1
        index = 1
        for y in range(0, self.size - 1):
            self.bridges[index].append(((y, x, index), (y, x - 1, self.from2D(y, x - 1)),
                                        (y + 1, x - 1, self.from2D(y + 1, x - 1)),
                                        (y + 1, x - 2, self.from2D(y + 1, x - 2))))
            index += self.size

    def valid_coords(self, y: int, x: int) -> bool:  # verifies that coordinates are within the board
        return 0 <= x < self.size and 0 <= y < self.size

    def valid_bridge_coords(self, y: int, x: int) -> bool:  # verifies that coordinates are within the board
        return -1 <= x < self.size + 1 and -1 <= y < self.size + 1

    def new_state(self):
        return [(0, 0) for i in range(self.size ** 2)] + [(1, 0)]

    def get_actions(self, state, known_not_endstate=False):
        return [i for i, t in enumerate(state[:-1]) if t[0] == t[1]] if known_not_endstate or not self.in_end_state(state) else []

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
        visits = defaultdict(lambda: False)
        if state[-1][0] == 1:
            stack = [x for x in range(0, self.size ** 2, self.size) if state[x][1] == 1]
            goal_test = (lambda x: x % self.size == self.size - 1)
            e = self.adjacencies_xsort
            c_index = 1
        else:
            stack = [x for x in range(self.size) if state[x][0] == 1]
            goal_test = (lambda x: x >= self.size * (self.size - 1))
            e = self.adjacencies_ysort
            c_index = 0
        while len(stack) > 0:
            v = stack.pop()
            if not visits[v]:
                if goal_test(v):
                    return True
                visits[v] = True
                for c in e[v]:
                    if state[c][c_index] == 1 and not visits[c]:
                        stack.append(c)
        return False

    def bfs(self, state):
        found = defaultdict(lambda: False)
        parents = defaultdict(lambda: -1)
        if state[-1][0] == 1:
            queue = [x for x in range(0, self.size ** 2, self.size) if state[x][1] == 1]
            goal_test = (lambda x: x % self.size == self.size - 1)
            e = self.adjacencies_xsort
            c_index = 1
        else:
            queue = [x for x in range(self.size) if state[x][0] == 1]
            goal_test = (lambda x: x >= self.size * (self.size - 1))
            e = self.adjacencies_ysort
            c_index = 0
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
            for c in e[v]:
                if state[c][c_index] == 1 and not found[c]:
                    parents[c] = v
                    found[c] = True
                    queue.append(c)
        return []

    def winner(self, state, known_endstate=False):
        return tuple(reversed(state[-1])) if known_endstate or self.in_end_state(state) else None

    def p1_reward(self, state, known_endstate=False):
        return state[-1][1] if known_endstate or self.in_end_state(state) else None

    def p1_to_move(self, state):
        return state[-1][0] == 1

    def visualize(self, states, player_labels=("Player 1", "Player 2"), use_board_labels=False):  # visualize states (list of states)
        G = nx.Graph()
        for i in range(self.size ** 2):
            y, x = self.from1D(i)
            pos = (x - 0.5 * (x + y), -x - y)  # use manhattan distance to find node positions
            G.add_node(i, pos=pos)
            for node in self.adjacencies[i]:  # use adjacency matrix to add edges
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
            if use_board_labels:
                nx.draw_networkx_labels(G, pos, font_weight="bold")  # draw node names (their coordinate)
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

    def complete_action_dist(self, action_dist: dict):
        vals = [0] * self.size ** 2
        for key, val in iter(action_dist.items()):
            vals[key] = val
        return vals

    def to_array(self, state):
        shape = (self.size + 2, self.size + 2)
        player1_array = np.zeros(shape=shape, dtype=float32)
        player1_array[0] = np.ones(shape=(self.size + 2))
        player1_array[0][0] = 0
        player1_array[0][self.size + 1] = 0
        player1_array[self.size + 1] = np.ones(shape=(self.size + 2))
        player1_array[self.size + 1][0] = 0
        player1_array[self.size + 1][self.size + 1] = 0
        player2_array = np.transpose(np.copy(player1_array))
        p1_to_move_array = np.zeros(shape=shape, dtype=float32) if state[-1][0] == 0 else np.ones(shape=shape, dtype=float32)
        p2_to_move_array = np.zeros(shape=shape, dtype=float32) if state[-1][0] == 1 else np.ones(shape=shape, dtype=float32)
        p1_bridge_ends = np.zeros(shape=shape, dtype=float32)
        p2_bridge_ends = np.zeros(shape=shape, dtype=float32)
        in_move_create_bridge = np.zeros(shape=shape, dtype=float32)
        in_move_complete_bridge = np.zeros(shape=shape, dtype=float32)
        in_move_save_bridge = np.zeros(shape=shape, dtype=float32)
        in_move_deny_bridge = np.zeros(shape=shape, dtype=float32)
        in_move_hurt_bridge = np.zeros(shape=shape, dtype=float32)
        in_move_kill_bridge = np.zeros(shape=shape, dtype=float32)
        in_move_kill_unbuilt_bridge = np.zeros(shape=shape, dtype=float32)
        in_move = state[-1]
        for i in range(1, self.size + 1):
            for j in range(1, self.size + 1):
                index = self.from2D(i - 1, j - 1)
                start_tuple = state[index]
                player1_array[i][j] = start_tuple[0]
                player2_array[i][j] = start_tuple[1]
                for (start_y, start_x, start_i), (path1_y, path1_x, path1_i),\
                    (path2_y, path2_x, path2_i), (end_y, end_x, end_i) in self.bridges[index]:
                    if not (state[path1_i] == (0, 0) or state[path2_i] == (0, 0)):  # both paths taken -> nothing to do
                        continue
                    end_owner = [0, 0]
                    if self.valid_coords(end_y, end_x):  # bridge ends inside board
                        end_owner = state[end_i]
                    else: # bridge ends outside board -> find owner
                        padding_coord_x = end_x + 1
                        padding_coord_y = end_y + 1
                        end_owner[0] = int(player1_array[padding_coord_y][padding_coord_x])
                        end_owner[1] = int(player2_array[padding_coord_y][padding_coord_x])
                        end_owner = tuple(end_owner)
                    if not (end_owner == state[start_i] or end_owner == (0, 0) or start_tuple == (0, 0)):
                        continue
                    if start_tuple == (0, 0):  # bridge origin empty
                        if end_owner == (0, 0):  # bridge dest and origin empty -> nothing to do
                            continue
                        if end_owner == in_move:  # in move has end, origin empty
                            if state[path1_i] == (0, 0):
                                if state[path2_i] == (0, 0) or state[path2_i] == in_move:  # in move has end and both paths unblocked, path1 open -> can create bridge at origin
                                    in_move_create_bridge[start_y + 1][start_x + 1] = 1
                                # else:  # in move has end path1 open, path 2 owned by not in move -> nothing to do
                                #    pass
                            elif state[path1_i] == in_move:  # origin empty, but end owned by in move and no blocked paths (only path2 open) -> create bridge at origin
                                in_move_create_bridge[start_y + 1][start_x + 1] = 1
                            #else: # in move has end, origin empty, path1 owned by not in move -> nothing to do
                            #    pass
                        else:  # not in move has end, origin empty
                            if state[path1_i] == (0, 0):
                                if state[path2_i] == (0, 0):  # not in move has end, origin empty, and both paths open -> deny bridge at origin
                                    in_move_deny_bridge[start_y + 1][start_x + 1] = 1
                                elif state[path2_i] == in_move: # not in move has end, origin empty, path2 owned by in_move -> kill unbuilt bridge at path1
                                    in_move_kill_unbuilt_bridge[path1_y + 1][path1_x + 1] = 1
                            elif state[path1_i] == in_move: # not in move owns origin, but in move owns path 1
                                # not in move has end and path2 must be open -> kill unmade bridge at path2
                                in_move_kill_unbuilt_bridge[path2_y + 1][path2_x + 1] = 1
                            # else:  # not in move owns end and path1, origin empty -> nothing to do
                            #    pass
                    elif start_tuple == in_move: # in_move owns bridge origin
                        if end_owner == (0, 0):
                            if state[path1_i] == (0, 0):
                                if state[path2_i] == (0, 0) or state[path2_i] == in_move: # in move owns origin, no blocked paths, and end open -> create bridge at end
                                    in_move_create_bridge[end_y + 1][end_x + 1] = 1
                            elif state[path1_i] == in_move:
                                # in move owns origin, end open, and we know path 2 open, in move owns path 1 -> create bridge at end
                                in_move_create_bridge[end_y + 1][end_x + 1] = 1
                            #else: # in move owns origin, end open and one path owned by not in move -> nothing to do
                            #    pass
                        else:  # in move has origin and end
                            if in_move == (1, 0): # in move has end and origin and we know at least one path open -> add bridge ends
                                p1_bridge_ends[start_y + 1][start_x + 1] = 1
                                p1_bridge_ends[end_y + 1][end_x + 1] = 1
                            else:
                                p2_bridge_ends[start_y + 1][start_x + 1] = 1
                                p2_bridge_ends[end_y + 1][end_x + 1] = 1
                            if state[path1_i] == (0, 0): # in move has origin, end and path1 open -> complete bridge at path1
                                in_move_complete_bridge[path1_y + 1][path1_x + 1] = 1
                                if state[path2_i] == (0, 0) or state[path2_i] == in_move: # in move has origin, end and path2 open -> complete bridge at path2
                                    in_move_complete_bridge[path2_y + 1][path2_x + 1] = 1
                                else: # in move has origin and end, and path1 open but path2 closed -> save bridge at path1
                                    in_move_save_bridge[path1_y + 1][path1_x + 1] = 1
                            elif state[path1_i] == in_move: # in move has origin, end and path 1
                                # path 2 must be open -> complete bridge at path 2
                                in_move_complete_bridge[path2_y + 1][path2_x + 1] = 1
                            else: # in move has origin and end, but path 1 taken, save and complete bridge at path 2 (it must be open)
                                in_move_complete_bridge[path2_y + 1][path2_x + 1] = 1
                                in_move_save_bridge[path2_y + 1][path2_x + 1] = 1
                        # else: # in move has origin, not in move has end -> nothing to do, always skipped (by continue)
                        #    pass
                    else: # not in move owns bridge origin
                        if end_owner == (0, 0):
                            if state[path1_i] == (0, 0):
                                if state[path2_i] == (0, 0): # not in move owns bridge origin and both paths and end open - > deny bridge at end
                                    in_move_deny_bridge[end_y + 1][end_x + 1] = 1
                                elif state[path2_i] == in_move:  # end open, origin owned by not in move, path 2 owned by in move
                                    # kill unbuilt bridge at path 1
                                    in_move_kill_bridge[path1_y + 1][path1_x + 1] = 1
                                # else: end open, path1 and origin owned by not in move - > nothing to do
                                #    pass
                            elif state[path1_i] == in_move:  # end open, origin owned by not in move, path 1 owned by in move
                                # kill unbuilt bridge at path2 (which must be open)
                                in_move_kill_unbuilt_bridge[path2_y + 1][path2_x + 1] = 1
                            # else: origin and path_1 owned by not in move, end and path2 open - > nothing to do
                            #    pass
                        #elif end_owner == in_move:  # in move has end, not in move has origin -> skipped by continue and nothing to do
                        #    pass
                        else:  # not in move has end and origin -> we know at least one path open, add bridge ends for not in move
                            if in_move == (1, 0):
                                p2_bridge_ends[start_y + 1][start_x + 1] = 1
                                p2_bridge_ends[end_y + 1][end_x + 1] = 1
                            else:
                                p1_bridge_ends[start_y + 1][start_x + 1] = 1
                                p1_bridge_ends[end_y + 1][end_x + 1] = 1
                            if state[path1_i] == (0, 0):
                                if state[path2_i] == (0, 0): # not in move owns end and origin, both paths open -> hurt bridge at both paths
                                    in_move_hurt_bridge[path1_y + 1][path1_x + 1] = 1
                                    in_move_hurt_bridge[path2_y + 1][path2_x + 1] = 1
                                elif state[path2_i] == in_move: # not in move owns end and origin, path 2 owned by in move -> kill bridge at path 1
                                    in_move_hurt_bridge[path1_y + 1][path1_x + 1] = 1
                                    in_move_kill_bridge[path1_y + 1][path1_x + 1] = 1
                                # else: not in move owns end origin and path 2 -> nothing to do
                                #    pass
                            elif state[path1_i] == in_move: # not in move owns end and origin, path1 owned by in move -> kill bridge at path2
                                in_move_hurt_bridge[path2_y + 1][path2_x + 1] = 1
                                in_move_kill_bridge[path2_y + 1][path2_x + 1] = 1
                            # else: # not in move owns end, origin, and path1 -> nothing to do
                            #    pass
        empty_array = np.ones(shape=(self.size + 2)) - player1_array - player2_array
        empty_array[0] = np.zeros(shape=(self.size + 2))
        empty_array[self.size + 1] = np.zeros(shape=(self.size + 2))
        arrays = np.array([
            empty_array,
            player1_array,
            player2_array,
            p1_to_move_array,
            p2_to_move_array,
            p1_bridge_ends,
            p2_bridge_ends,
            in_move_create_bridge,
            in_move_complete_bridge,
            in_move_save_bridge,
            in_move_deny_bridge,
            in_move_hurt_bridge,
            in_move_kill_bridge,
            in_move_kill_unbuilt_bridge,
        ])
        return np.transpose(arrays, axes=[1, 2, 0])


    def augment_training_data(self, state_array, action_dist):
        states = [state_array, np.flip(state_array, axis=[0, 1])]
        dists = [action_dist, action_dist[::-1]]
        return states, dists

    def array_to_state(self, array):
        count = 0
        state = [None] * self.size ** 2
        print(array)
        for y in range(self.size):
            for x in range(self.size):
                val = [0, 0]
                if array[y][x][0] != 0:
                    count += 1
                    val[0 if int(array[y][x][0]) == 1 else 1] = 1
                state[self.from2D(y, x)] = tuple(val)
        state.append((1, 0) if count % 2 == 0 else (0, 1))
        return state

if __name__ == "__main__":
    cfg = {
        "size": 5
    }
    game = HexWorld(cfg, 0.3)

    states = []
    state = game.new_state()
    actions = game.get_actions(state)
    print(np.transpose(game.to_array(state), axes=[2, 0, 1]))
    while len(actions) > 0:
        state = game.do_action(state, actions[random.randint(0, len(actions) - 1)])
        actions = game.get_actions(state)
        states.append(state)

    #print(game.to_array(states[-2]))

    game.visualize([states[-2]])




