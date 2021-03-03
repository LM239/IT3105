import getopt
import sys
import yaml
from interfaces.world import SimWorld
from worlds.hex_world import HexWorld
from worlds.nim_world import NimWorld
from anet import Anet
from configs.validate_configs import validate_topp_config
from actor import Actor
from collections import defaultdict
import glob
import os


class Topp():
    def __init__(self, topp_cfg, state_manager: SimWorld):
        validate_topp_config(topp_cfg)
        self.games_g = topp_cfg["games_g"]
        self.display_games_pairs = []
        self.state_manager = state_manager
        self.display_games = topp_cfg["display_games"]
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        if "display_games_pairs" in topp_cfg:
            self.display_games_pairs = topp_cfg["display_games_pairs"]

        files = glob.glob(topp_cfg["directory"] + "*.h5")
        print(files)
        self.actors = {file[file.rindex("\\")+1:]: Actor(Anet(model_file=file), self.state_manager) for file in files}

        self.total_wins = defaultdict(lambda: 0)


    def run_tour(self):
        for i, file1 in enumerate(list(self.actors.keys())):
            for j, file2 in enumerate(list(self.actors.keys())[i+1:]):
                print(file1, "is competing with", file2)
                self.compete(file1, file2)
        rankings = sorted([(self.total_wins[key], key) for key in self.total_wins.keys()], reverse=True)
        for i, rank in enumerate(rankings):
            losses = (len(self.actors)-1)*self.games_g-rank[0]
            print(str(i+1) + ".", "Checkpoint", rank[1], "with", rank[0], "wins and", losses, "losses")

    def compete(self, actor1_key: str, actor2_key: str):
        actor1, actor2 = (self.actors[actor1_key], self.actors[actor2_key])
        display_game = False
        for pair in self.display_games_pairs:
            if actor1_key in pair and actor2_key in pair and self.display_games:
                display_game = True
        for i in range(self.games_g):
            states = []
            state = self.state_manager.new_state()
            states.append(state)
            move = i % 2
            while not self.state_manager.in_end_state(state):
                if move % 2 == 0:
                    action = actor1.get_move(state)
                    state = self.state_manager.do_action(state, action)
                else:
                    action = actor2.get_move(state)
                    state = self.state_manager.do_action(state, action)
                states.append(state)
                move += 1
            winner = self.state_manager.winner(state)
            if display_game:
                player_labels = (actor1_key, actor2_key) if i % 2 == 0 else (actor2_key, actor1_key)
                self.state_manager.visualize(states, player_labels=player_labels)
            if i % 2 == 0:
                if winner[0] == 1:
                    self.total_wins[actor1_key] += 1
                else:
                    self.total_wins[actor2_key] += 1
            else:
                if winner[0] == 1:
                    self.total_wins[actor2_key] += 1
                else:
                    self.total_wins[actor1_key] += 1






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
