def validate_config(configs):
    if "display" not in configs:
        print("Missing required argument 'display' in config \n Exiting")
        exit(1)
    if "sim_world" not in configs:
        print("Missing sim_world dict in config \n Exiting")
        exit(1)
    if "mcts" not in configs:
        print("Missing mcts dict in config \n Exiting")
        exit(1)
    if "anet" not in configs:
        print("Missing anet dict in config \n Exiting")
        exit(1)
    if "topp" not in configs:
        print("Missing topp dict in config \n Exiting")
        exit(1)
    return True

def validate_nim(config):
    if "n" not in config:
        print("Missing required nim argument: 'n' \nExiting")
        exit(1)
    if "k" not in config:
        print("Missing required nim argument: 'k' \nExiting")
        exit(1)

def validate_hex_board(config):
    if "size" not in config:
        print("Missing required hex argument: 'size' \nExiting")
        exit(1)
    elif config["size"] < 3 or config["size"] > 10:
        print("Size parameter outside of accepted range: [3,) \nExiting")
        exit(1)


def validate_mcts(config):
    if "episodes" not in config:
        print("Missing required mcts argument: 'episodes' \nExiting")
        exit(1)
    if "search_duration" not in config:
        print("Missing required actor_config argument: 'search_duration' \nExiting")
        exit(1)
    if "bias" not in config:
        print("Missing required actor_config argument: 'bias' \nExiting")
        exit(1)


def validate_anet(config):
    valid_activations = ["linear", "sigmoid", "tanh", "relu"]
    valid_optmizers = ["Adagrad", "SGD", "RMSProp", "Adam"]
    if "lr" not in config:
        print("Missing required anet argument: 'lr' \nExiting")
        exit(1)
    if "optimizer" not in config:
        print("Missing required anet argument: 'optimizer' \nExiting")
        exit(1)
    if "hidden_layers" not in config:
        print("Missing required anet argument: 'hidden_layers' \nExiting")
        exit(1)
    if config["optimizer"] not in valid_optmizers:
        print("ANET optimizer invalid. Valid options are:", valid_optmizers, "\nExiting")
    for i, layer in enumerate(config["hidden_layers"]):
        if layer[1] not in valid_activations:
            print("ANET layer", i, "activation function is invalid. Valid options are:", valid_activations, "\nExiting")


def validate_topp(config):
    if "cache_m" not in config:
        print("Missing required topp argument: 'cache_m' \nExiting")
        exit(1)
    if "games_g" not in config:
        print("Missing required topp argument: 'games_g' \nExiting")
        exit(1)
