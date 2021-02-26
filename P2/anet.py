from configs.validate_configs import validate_anet_config
from interfaces.actornet import ActorNet
from keras.models import Sequential
from keras.layers import Dense, Input
import tensorflow as tf
import os


class Anet(ActorNet):
    def __init__(self, anet_cfg, input_dim: int, output_dim: int):
        validate_anet_config(anet_cfg)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        self.size = anet_cfg["hidden_layers"]
        self.lr = anet_cfg["lr"]

        self.file_structure = anet_cfg["file_structure"]

        # Build the network with first layer of input size
        # and hidden layers with sizes from config
        self.model = Sequential()
        self.model.add(Input(shape=(input_dim,)))
        for layer in self.size:
            self.model.add(Dense(layer[0], activation=layer[1]))
        self.model.add(Dense(output_dim, activation=anet_cfg["output_activation"]))
        self.model.compile(optimizer=anet_cfg["optimizer"], loss=anet_cfg["loss"])

    def train(self, features, targets):
        self.model.fit(
            features,  # training data
            targets,  # training targets
            epochs=1
        )

    def forward(self, input):
        return self.model(tf.convert_to_tensor([input]), training=False)

    def save_params(self, episode: int):
        file_name = self.file_structure + "checkpoint_" + episode
        self.model.save(file_name)
