import getopt
import sys
import yaml
from worlds.hex_world import HexWorld
from anet import Anet
from configs.validate_configs import validate_topp_config
from actor import Actor

class Topp():
    def __init__(self, topp_cfg):
        validate_topp_config(topp_cfg)
        self.games_g = topp_cfg["games_g"]
        self.display_games = []
        if "display_games" in topp_cfg:
            self.display_games = topp_cfg["display_games"]

    def run_tour(self):
        pass


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

    topp = Topp(topp_config["topp"])