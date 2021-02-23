import getopt
import sys
import yaml
from worlds.hex_world import HexWorld
from worlds.nim_world import NimWorld
from mc_rave import McRave
from configs.validate_configs import validate_config


if __name__ == "__main__":
    short_options = "h"
    long_options = ["help"]  # command line options use either python run.py -h | --help

    try:
        arguments, values = getopt.getopt(sys.argv[1:], short_options, long_options)  # set up cmd options
    except getopt.error as err:
        print(str(err))
        sys.exit(2)

    for current_argument, current_value in arguments:  # check options
        if current_argument in ("-h", "--help"):
            print("Usage: python {} <path-to-config>".format(sys.argv[0]))  # print help
            exit(0)

    if not len(sys.argv) == 2:
        print("Error: Expected 1 argument <path-to-config>, but received {} argument(s)\nExiting".format(len(sys.argv) - 1))
        exit(1)

    try:
        with open(sys.argv[1]) as file:  # try to open config file
            configs = yaml.load(file, Loader=yaml.FullLoader)
            file.close()
    except FileNotFoundError:
        print("Could not find file at {}\nExiting".format(sys.argv[1]))
        exit(1)

    validate_config(configs)  # do an initial validation of the configs (not thorough)
    mcts_config = configs["mcts"]
    anet_config = configs["anet"]
    world_config = configs["sim_world"]

    if "display_episodes" in configs:  # create list of episodes to visualize
        display_episodes = configs["display_episodes"]
    else:
        display_episodes = []

    node_heuristic = None
    if world_config["world"] == "hex":  # create sim_world for the actor critic
        world_manager = HexWorld(world_config)
        node_heuristic = (lambda state: (50 * sum(1 for t in state[:-1] if t[0] == t[1]) / (len(state) - 1) ))

        mcts = McRave(mcts_config, world_manager, node_heuristic)
    elif world_config["world"] == "nim":
        world_manager = NimWorld(world_config)
        mcts = McRave(mcts_config, world_manager)
        mcts.run_root(world_manager.new_state())
    else:
        print("Unknown world type: {} \n Exiting".format(world_config["world"]))
        exit(1)

    exit(0)
