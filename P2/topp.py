import getopt
import sys
import yaml
from interfaces.world import SimWorld
from worlds.hex_world import HexWorld
from worlds.nim_world import NimWorld
from configs.validate_configs import validate_topp_config
from players import GreedyPlayer, ProbabilisticPlayer, Player
from collections import defaultdict
from conv_anet import ConvNet
import os
import glob


def compete(player1: Player, player2: Player, num_games: int, state_manager: SimWorld, display_game=0):
    player1_wins = 0
    player2_wins = 0
    for i in range(num_games):
        states = []
        state = state_manager.new_state()
        states.append(state)
        move = i % 2
        while not state_manager.in_end_state(state):
            if move % 2 == 0:
                action = player1.play(state)
                state = state_manager.do_action(state, action)
            else:
                action = player2.play(state)
                state = state_manager.do_action(state, action)
            states.append(state)
            move += 1
        winner = state_manager.winner(state, known_endstate=True)
        if i < display_game:
            player_labels = (player1.name, player2.name) if i % 2 == 0 else (player2.name, player1.name)
            state_manager.visualize(states, player_labels=player_labels)
        if i % 2 == 0:
            if winner[0] == 1:
                player1_wins += 1
            else:
                player2_wins += 1
        else:
            if winner[0] == 1:
                player2_wins += 1
            else:
                player1_wins += 1
    return player1_wins, player2_wins



class Topp():
    def __init__(self, topp_cfg, state_manager: SimWorld):
        validate_topp_config(topp_cfg)
        self.games_g = topp_cfg["games_g"]
        self.display_games_pairs = []
        self.state_manager = state_manager
        self.display_games = topp_cfg["display_games"]

        self.player_type = GreedyPlayer if topp_cfg["player_type"] == "greedy" else ProbabilisticPlayer
        if "display_games_pairs" in topp_cfg:
            self.display_games_pairs = [tuple(sorted(p)) for p in topp_cfg["display_games_pairs"]]

        files = glob.glob(topp_cfg["directory"] + "*.h5")
        print(files)
        self.players = None
        if topp_cfg["player_type"] == "greedy":
            self.players = {file[file.rindex("\\") + 1:]: GreedyPlayer(ConvNet(model_file=file), self.state_manager, file[file.rindex("\\") + 1:]) for file in files}
        else:
            self.players = {file[file.rindex("\\") + 1:]: ProbabilisticPlayer(ConvNet(model_file=file), self.state_manager,
                                                                       file[file.rindex("\\") + 1:]) for file in files}
        self.total_wins = defaultdict(lambda: 0)

    def run_tour(self):
        for i, player1 in enumerate(self.players.values()):
            for j, player2 in enumerate(list(self.players.values())[i + 1:]):
                print(player1.name, "is competing with", player2.name)
                player1_wins, player2_wins = compete(player1, player2, self.games_g, self.state_manager, self.display_games if (player1.name, player2.name) in self.display_games_pairs else 0)
                self.total_wins[player1.name] += player1_wins
                self.total_wins[player2.name] += player2_wins
        rankings = sorted([(self.total_wins[key], key) for key in self.total_wins.keys()], reverse=True)
        for i, rank in enumerate(rankings):
            losses = (len(self.players) - 1) * self.games_g - rank[0]
            print(str(i+1) + ".", "Checkpoint", rank[1], "with", rank[0], "wins and", losses, "losses")


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

    if "player_type" not in topp_config["topp"]:
        print("Missing required config value 'player_type' \nExiting")
        exit(1)

    if topp_config["topp"]["player_type"] not in ["probabilistic", "greedy"]:
        print("Unknown player type {}, valid types are: {} \nExiting".format(topp_config["player_type"], ["probabilistic", "greedy"]))
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

    if "allow_cuda" in topp_config and not topp_config["allow_cuda"]:
        print("Disable cuda")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    topp = Topp(topp_config["topp"], world_manager)
    topp.run_tour()
