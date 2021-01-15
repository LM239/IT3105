def validate_config(configs):
    if "episodes" not in configs:
        print("Missing required argument 'episodes' in config \n Exiting")
        exit(1)
    if "actor" not in configs:
        print("Missing actor dict in config \n Exiting")
        exit(1)
    if "critic" not in configs:
        print("Missing critic dict in config \n Exiting")
        exit(1)
    if "sim_world" not in configs:
        print("Missing sim_world dict in config \n Exiting")
        exit(1)
    if "episodes" not in configs:
        print("Missing parameter 'episodes' in config \n Exiting")
        exit(1)
    if "world" not in configs["sim_world"]:
        print("Missing required argument 'world' in sim_world config \n Exiting")
        exit(1)

def validate_pegsol_config(config):
    if "type" not in config:
        print("Missing required PegSolitaire argument: 'type' \nExiting")
        exit(1)
    if "size" not in config:
        print("Missing required PegSolitaire argument: 'size' \nExiting")
        exit(1)
    elif config["size"] < 3:
        print("Size parameter too small \nExiting")
        exit(1)
    if not (config["type"] == "triangle" or config["type"] == "diamond"):
        print("Unknown board type {}.\nExiting".format(config["type"]))
        exit(1)


def validate_actor_critic_config(actor_config, critic_config):
    if "lr" not in actor_config:
        print("Missing required actor_config argument: 'lr' \nExiting")
        exit(1)
    if "eligibility_decay" not in actor_config:
        print("Missing required actor_config argument: 'eligibility_decay' \nExiting")
        exit(1)
    if "discount_factor" not in actor_config:
        print("Missing required actor_config argument: 'discount_factor' \nExiting")
        exit(1)
    if "greedy_epsilon" not in actor_config:
        print("Missing required actor_config argument: 'greedy_epsilon' \nExiting")
        exit(1)
    if "greedy_epsilon" not in actor_config:
        print("Missing required actor_config argument: 'greedy_epsilon' \nExiting")
        exit(1)
    if "epsilon_decay" not in actor_config:
        print("Missing required actor_config argument: 'epsilon_decay' \nExiting")
        exit(1)

    if "type" not in critic_config:
        print("Missing required Critic argument: 'type' \nExiting")
        exit(1)
    if not (critic_config["type"] == "neural_net" or critic_config["type"] == "table"):
        print("Unknown critic type: {} \nExiting".format(critic_config["type"]))
        exit(1)
    if critic_config["type"] == "neural_net" and "size" not in critic_config:
        print("Missing required neural net based-Critic argument: 'size' \nExiting")
        exit(1)
    if "lr" not in critic_config:
        print("Missing required Critic argument: 'lr' \nExiting")
        exit(1)
    if "eligibility_decay" not in critic_config:
        print("Missing required Critic argument: 'eligibility_decay' \nExiting")
        exit(1)
    if "discount_factor" not in critic_config:
        print("Missing required Critic argument: 'discount_factor' \nExiting")
        exit(1)
