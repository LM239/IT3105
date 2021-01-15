import matplotlib as plt


class Actor_critic:
    
    def __init__(self, actor_config, critic_config, world, episodes):
        self.verify_configs(actor_config, critic_config)

        self.episodes = episodes
        self.actor_lr = actor_config["lr"]
        self.actor_eligibility_decay = actor_config["eligibility_decay"]
        self.actor_discount_factor = actor_config["discount_factor"]
        self.actor_greedy_epsilon = actor_config["greedy_epsilon"]
        self.actor_greedy_epsilon = actor_config["greedy_epsilon"]
        self.actor_epsilon_decay = actor_config["epsilon_decay"]

        self.critic_type = critic_config["type"]
        self.critic_lr = critic_config["lr"]
        self.critic_eligibility_decay = critic_config["eligibility_decay"]
        self.critic_discount_factor = critic_config["discount_factor"]

        if self.critic_type == "neural_net":
            self.critic_size = critic_config["size"]

    def verify_configs(self, actor_config, critic_config):
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
        if critic_config["type"] == "neural_network" and "size" not in critic_config:
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

