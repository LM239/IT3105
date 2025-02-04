def validate_config(configs):
    if "sim_world" not in configs:
        print("Missing sim_world dict in config \n Exiting")
        exit(1)
    if "world" not in configs["sim_world"]:
        print("Missing required sim_world argument 'world' in world_config \n Exiting")
        exit(1)
    if "mcts" not in configs:
        print("Missing mcts dict in config \n Exiting")
        exit(1)
    if "anet" not in configs:
        print("Missing anet dict in config \n Exiting")
        exit(1)
    if "actor" not in configs:
        print("Missing actor dict in config \n Exiting")
        exit(1)


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
    elif config["size"] < 3:
        print("Size parameter outside of accepted range: [3,) \nExiting")
        exit(1)

def validate_actor_config(config):
    if "episodes" not in config:
        print("Missing required actor argument: 'episodes' \nExiting")
        exit(1)
    if "num_checkpoints" not in config:
        print("Missing required actor argument: 'num_checkpoints' \nExiting")
        exit(1)
    if "train_ex_size" not in config:
        print("Missing required actor argument: 'train_ex_size' \nExiting")
        exit(1)
    if "competition_games" not in config:
        print("Missing required actor argument: 'competition games' \nExiting")
        exit(1)
    if "anet_dir" not in config:
        print("Missing required anet argument: 'anet_dir' \nExiting")
        exit(1)
    if "save_data" not in config:
        print("Missing required anet argument: 'save_data' \nExiting")
        exit(1)
    if "win_margin" not in config:
        print("Missing required anet argument: 'win_margin' \nExiting")
        exit(1)

def validate_mcts_config(config):
    if "search_duration" not in config:
        print("Missing required actor_config argument: 'search_duration' \nExiting")
        exit(1)


def validate_anet_config(config):
    valid_activations = ["linear", "sigmoid", "tanh", "relu", "selu"]
    valid_optmizers = ["Adagrad", "SGD", "RMSProp", "adam"]
    if "lr" not in config:
        print("Missing required anet argument: 'lr' \nExiting")
        exit(1)
    if "optimizer" not in config:
        print("Missing required anet argument: 'optimizer' \nExiting")
        exit(1)
    if config["optimizer"] not in valid_optmizers:
        print("ANET optimizer invalid. Valid options are:", valid_optmizers, "\nExiting")
    if "activation" not in config:
        print("Missing required anet argument: 'activation' \nExiting")
        exit(1)
    if config["activation"] not in valid_activations:
        print("Activation function {} is invalid. Valid options are: {}\nExiting".format(config["activation"], valid_activations))
    if "cnn_filters" not in config:
        print("Missing required anet argument: 'cnn_filters' \nExiting")
        exit(1)
    if "dense_layers" not in config:
        print("Missing required anet argument: 'dense_layers' \nExiting")
        exit(1)


def validate_topp_config(config):
    if "games_g" not in config:
        print("Missing required topp argument: 'games_g' \nExiting")
        exit(1)
    if "directory" not in config:
        print("Missing required topp argument: 'directory' \nExiting")
        exit(1)