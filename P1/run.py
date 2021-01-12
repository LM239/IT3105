import getopt
import sys
import yaml
from worlds.pegsol_world import PegSolitaire

if __name__ == "__main__":
    short_options = "hc:"
    long_options = ["help", "config"]

    full_cmd_arguments = sys.argv
    argument_list = full_cmd_arguments[1:]  # remove filename

    try:
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
    except getopt.error as err:
        # Output error, and return with an error code
        print(str(err))
        sys.exit(2)

    path = ""
    for current_argument, current_value in arguments:
        if current_argument in ("-h", "--help"):
            print("Required args: \n Optional args:")
        elif current_argument in ("-c", "--config"):
            path = current_value

    try:
        with open(path) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)
            file.close()
    except FileNotFoundError:
       print("Could not open file at {}".format(path))
       print("Exiting")
       exit(1)

    actor_config = configs["actor"]
    critic_config = configs["critic"]
    world_config = configs["sim_world"]

    world = PegSolitaire(world_config)
    print(actor_config)
    print(critic_config)
    print(world_config)

    exit(0)
