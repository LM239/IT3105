import getopt
import sys
import yaml
from worlds.pegsol_world import PegSolitaire
from actor_critic import ActorCritic
from configs.validate_configs import validate_config
from table_critic import TableCritic

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
    actor_config = configs["actor"]
    critic_config = configs["critic"]
    world_config = configs["sim_world"]

    if "display_episodes" in configs:  # create list of episodes to visualize
        display_episodes = configs["display_episodes"]
    else:
        display_episodes = []

    if world_config["world"] == "peg_solitaire":  # create sim_world for the actor critic
        world = PegSolitaire(world_config)
    else:
        print("Unknown world type: {} \n Exiting".format(world_config["world"]))
        exit(1)

    if critic_config["type"] == "neural_net":  # create critic for actor critic
        from neural_critic import NeuralCritic
        world_size = world.size**2 if world.type == "diamond" else int(world.size * (world.size + 1) / 2)  # input layer dim, given by board size
        critic = NeuralCritic(critic_config, world_size)  # nn based
    else:
        critic = TableCritic(critic_config)   # table based

    actor_critic = ActorCritic(actor_config, critic, world, configs["episodes"], display_episodes)  # make actor
    actor_critic.fit()  # train actor
    
    if "display_greedy" in configs and configs["display_greedy"]:
        actor_critic.play_episode()  # visualize policy with epsilon=0
        world.visualize_episode()
    world.visualize_peg_count()  # show peg count

    exit(0)
