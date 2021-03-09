import getopt
import sys
import yaml
from interfaces.world import SimWorld
from worlds.hex_world import HexWorld
from worlds.nim_world import NimWorld
from players import ProbabilisticPlayer, HumanPlayer, GreedyPlayer
from conv_anet import ConvNet
import os


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
            challenge_cfg = yaml.load(file, Loader=yaml.FullLoader)
            file.close()
    except getopt.error as err:
        print(str(err))
        sys.exit(2)
    except FileNotFoundError:
        print("Could not find file at {}\nExiting".format(sys.argv[1]))
        exit(1)

    if "sim_world" not in challenge_cfg:
        print("Missing required config dict 'sim_world' \nExiting")
        exit(1)

    world_config = challenge_cfg["sim_world"]

    if world_config["world"] == "hex":  # create sim_world for the actor critic
        state_manager: SimWorld = HexWorld(world_config, 1)
    elif world_config["world"] == "nim":
        state_manager: SimWorld = NimWorld(world_config)
    else:
        print("Unknown world type: {} \n Exiting".format(world_config["world"]))
        exit(1)

    if "allow_cuda" in challenge_cfg and not challenge_cfg["allow_cuda"]:
        print("Disable cuda")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    player1 = HumanPlayer(state_manager)
    player2 = GreedyPlayer(ConvNet(model_file=challenge_cfg["opponent"]), state_manager, "Terminator")

    state = state_manager.new_state()
    move = (challenge_cfg["play_as"] - 1) % 2
    while not state_manager.in_end_state(state):
        if move % 2 == 0:
            action = player1.play(state)
            state = state_manager.do_action(state, action)
        else:
            action = player2.play(state)
            state = state_manager.do_action(state, action)
        move += 1

    remark = "Congratulations, you won!" if state_manager.winner(state, True)[challenge_cfg["play_as"] - 1] == 1 else "You lost, better luck next time!"
    print(remark)
    state_manager.visualize()


