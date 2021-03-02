from configs.validate_configs import validate_anet_config
from interfaces.actornet import ActorNet
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import load_model
import tensorflow as tf


class Anet(ActorNet):
    def __init__(self, anet_cfg=None, input_dim: int = 0, output_dim: int = 0, model_file: str = None):
        if model_file is None:
            validate_anet_config(anet_cfg)
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
            opt = type(tf.keras.optimizers.get(anet_cfg["optimizer"]))(learning_rate=anet_cfg["lr"])
            self.model.compile(optimizer=opt, loss=anet_cfg["loss"])
        else:
            self.model = load_model(model_file)

    def train(self, features, targets):
        self.model.fit(
            features,  # training data
            targets,  # training targets
            epochs=1
        )

    def forward(self, input):
        return self.model(tf.convert_to_tensor([input]), training=False)

    def save_params(self, episode: int):
        file_name = self.file_structure + "checkpoint_" + str(episode) + ".h5"
        self.model.save(file_name)

