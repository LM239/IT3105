import getopt
import sys
import yaml
from worlds.hex_world import HexWorld
from worlds.nim_world import NimWorld
from search.treesearch import hex_search
from mc_rave import McRave
from conv_anet import ConvNet
from configs.validate_configs import validate_config
from actor import StandardActor, TourActor
from shutil import copyfile


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

    try:
        copyfile(sys.argv[1], anet_config["file_structure"]+"config.yaml")
    except:
        "Failed to copy file"

    display_config = configs["display"] if "display" in configs else None
    display_rate = display_config["display_rate"] if "display_rate" in display_config else 0.2

    world_manager = None
    mcts = None
    anet = None

    if world_config["world"] == "hex":  # create sim_world for the actor critic

        world_manager = HexWorld(world_config, display_rate)

        input_dim = (world_config["size"]**2 + 1) * 2
        output_dim = world_config["size"]**2
        anet = ConvNet(anet_config, world_config["size"], output_dim)
        mcts = McRave(mcts_config, world_manager, anet, hex_search)
    elif world_config["world"] == "nim":
        world_manager = NimWorld(world_config)
        mcts = McRave(mcts_config, world_manager, anet)
    else:
        print("Unknown world type: {} \n Exiting".format(world_config["world"]))
        exit(1)

    if actor_config["type"] == "tour":
        actor = TourActor(anet, world_manager, actor_config, mcts)
    elif actor_config["type"] == "standard":
        actor = StandardActor(anet, world_manager, actor_config, mcts)
    actor.fit()
    exit(0)
