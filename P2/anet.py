from configs.validate_configs import validate_anet_config
from interfaces.actornet import ActorNet

class Anet(ActorNet):
    def __init__(self, anet_cfg, input_dim: int, output_dim: int):
        validate_anet_config(anet_cfg)

    def train(self, features, target):
        pass

    def forward(self, input):
        pass

    def save_params(self):
        pass