import getopt
import sys
import yaml
from worlds.pegsol_world import PegSolitaire
from actor_critic import ActorCritic
from configs.validate_configs import validate_config

if __name__ == "__main__":
    short_options = "h"
    long_options = ["help"]

    try:
        arguments, values = getopt.getopt(sys.argv[1:], short_options, long_options)
    except getopt.error as err:
        print(str(err))
        sys.exit(2)

    for current_argument, current_value in arguments:
        if current_argument in ("-h", "--help"):
            print("Usage: python {} <path-to-config>".format(sys.argv[0]))
            exit(0)

    try:
        path = sys.argv[1]
    except IndexError:
        print("Error: No path specified\nExiting")
        exit(1)

    try:
        with open(path) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)
            file.close()
    except FileNotFoundError:
        print("Could not find file at {}\nExiting".format(path))
        exit(1)

    validate_config(configs)
    actor_config = configs["actor"]
    critic_config = configs["critic"]
    world_config = configs["sim_world"]

    if "display_episodes" in configs:
        display_episodes = configs["display_episodes"]
    else:
        display_episodes = []

    if world_config["world"] == "peg_solitaire":
        world = PegSolitaire(world_config)
    else:
        print("Unknown world type: {} \n Exiting".format(world_config["world"]))
        exit(1)

    actor_critic = ActorCritic(actor_config, critic_config, world, configs["episodes"], display_episodes)
    actor_critic.fit()
    actor_critic.play_episode()
    world.visualize_peg_count()

    exit(0)
