import getopt
import sys
import yaml
from interfaces.world import SimWorld
from worlds.hex_world import HexWorld
from worlds.nim_world import NimWorld
from anet import Anet
from configs.validate_configs import validate_topp_config
from actor import Actor
import glob
import numpy as np


class Topp():
    def __init__(self, topp_cfg, state_manager: SimWorld):
        validate_topp_config(topp_cfg)
        self.games_g = topp_cfg["games_g"]
        self.display_games = []
        self.state_manager = state_manager
        if "display_games" in topp_cfg:
            self.display_games = topp_cfg["display_games"]

        self.actors = []
        files = glob.glob(topp_cfg["directory"] + "*.h5")
        print(files)
        for file in files:
            anet_model = Anet(model_file=file)
            self.actors.append(Actor(anet_model, self.state_manager))

        self.results = [[0 for i in range(len(self.actors))] for j in range(len(self.actors))]

    def run_tour(self):
        for i in range(len(self.actors)):
            for j in range(i+1, len(self.actors)):
                print(i, "is competing with", j)
                self.compete(i, j)
        print(np.array(self.results))

    def compete(self, actor1_index: int, actor2_index: int):
        actor1, actor2 = (self.actors[actor1_index], self.actors[actor2_index])
        for i in range(self.games_g):
            state = self.state_manager.new_state()
            move = i % 2
            while not self.state_manager.in_end_state(state):
                if move % 2 == 0:
                    action = actor1.get_move(state)
                    print(action)
                    state = self.state_manager.do_action(state, action)
                else:
                    action = actor2.get_move(state)
                    print(action)
                    state = self.state_manager.do_action(state, action)
                move += 1
            winner = self.state_manager.winner(state)
            if i % 2 == 0:
                if winner[0] == 1:
                    self.results[actor1_index][actor2_index] += 1
                else:
                    self.results[actor2_index][actor1_index] += 1
            else:
                if winner[0] == 1:
                    self.results[actor2_index][actor1_index] += 1
                else:
                    self.results[actor1_index][actor2_index] += 1






if __name__ == "__main__":
    short_options = "h"
    long_options = ["help"]  # command line options use either python run.py -h | --help

    try:
        arguments, values = getopt.getopt(sys.argv[1:], short_options, long_options)  # set up cmd options
        for current_argument, current_value in arguments:  # check options
            if current_argument in ("-h", "--help"):
                print("Usage: python {} <path-to-config>".format(sys.argv[0]))  # print help
                exit(0)

        if not len(sys.argv) == 2:
            print("Error: Expected 1 argument <path-to-config>, but received {} argument(s)\nExiting".format(
                len(sys.argv) - 1))
            exit(1)

        with open(sys.argv[1]) as file:  # try to open config file
            topp_config = yaml.load(file, Loader=yaml.FullLoader)
            file.close()
    except getopt.error as err:
        print(str(err))
        sys.exit(2)
    except FileNotFoundError:
        print("Could not find file at {}\nExiting".format(sys.argv[1]))
        exit(1)

    if "topp" not in topp_config:
        print("Missing required config dict 'topp' \nExiting")
        exit(1)

    if "sim_world" not in topp_config:
        print("Missing required config dict 'sim_world' \nExiting")
        exit(1)

    if "display_rate" not in topp_config:
        print("Missing required config value 'display_rate' \nExiting")
        exit(1)

    world_config = topp_config["sim_world"]
    display_rate = topp_config["display_rate"]

    world_manager = None

    if world_config["world"] == "hex":  # create sim_world for the actor critic
        world_manager = HexWorld(world_config, display_rate)
    elif world_config["world"] == "nim":
        world_manager = NimWorld(world_config)
    else:
        print("Unknown world type: {} \n Exiting".format(world_config["world"]))
        exit(1)

    topp = Topp(topp_config["topp"], world_manager)
    topp.run_tour()
