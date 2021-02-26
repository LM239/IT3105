import getopt
import sys
import yaml
from worlds.hex_world import HexWorld
from worlds.nim_world import NimWorld
from search.treesearch import hex_search
from mc_rave import McRave
from anet import Anet
from configs.validate_configs import validate_config
from actor import Actor


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
            print("Error: Expected 1 argument <path-to-config>, but received {} argument(s)\nExiting".format(len(sys.argv) - 1))
            exit(1)

        with open(sys.argv[1]) as file:  # try to open config file
            configs = yaml.load(file, Loader=yaml.FullLoader)
            file.close()
    except getopt.error as err:
        print(str(err))
        sys.exit(2)
    except FileNotFoundError:
        print("Could not find file at {}\nExiting".format(sys.argv[1]))
        exit(1)

    validate_config(configs)  # do an initial validation of the configs (not thorough)
    mcts_config = configs["mcts"]
    anet_config = configs["anet"]
    world_config = configs["sim_world"]
    actor_config = configs["actor"]

    display_config = configs["display"] if "display" in configs else None
    display_rate = display_config["display_rate"] if "display_rate" in display_config else 0.2

    node_heuristic = None
    world_manager = None
    mcts = None
    anet = None
    if world_config["world"] == "hex":  # create sim_world for the actor critic
        world_manager = HexWorld(world_config, display_rate)

        input_dim = (world_config["size"]**2 + 1) * 2
        output_dim = world_config["size"]
        anet = Anet(anet_config, input_dim, output_dim)
        node_heuristic = (lambda: 3)
        mcts = McRave(mcts_config, world_manager, anet, node_heuristic, hex_search)
    elif world_config["world"] == "nim":
        world_manager = NimWorld(world_config)
        mcts = McRave(mcts_config, world_manager, anet)
    else:
        print("Unknown world type: {} \n Exiting".format(world_config["world"]))
        exit(1)
    actor = Actor(actor_config, anet, mcts, world_manager)
    actor.fit()

    exit(0)
