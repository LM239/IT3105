from configs.validate_configs import validate_anet_config
from interfaces.actornet import ActorNet
from keras.models import Sequential
from keras.layers import Dense, Input

class Anet(ActorNet):
    def __init__(self, anet_cfg, input_dim: int, output_dim: int):
        validate_anet_config(anet_cfg)

        self.size = anet_cfg["hidden_layers"]
        self.lr = anet_cfg["lr"]

        self.file_structure =

        # Build the network with first layer of input size, last layer with size 1,
        # and hidden layers with sizes from config
        self.model = Sequential()
        self.model.add(Input(shape=(input_dim,)))
        for layer in self.size:
            self.model.add(Dense(layer[0], activation=layer[1]))
        self.model.add(Dense(output_dim, activation=anet_cfg["output_activation"]))
        self.model.compile(optimizer=anet_cfg["optimizer"], loss=anet_cfg["loss"])

    def train(self, features, target):
        self.model.fit(
            features,  # training data
            target,  # training targets
            epochs=1
        )

    def forward(self, input):
        return self.model.predict([input], batch_size=1)

    def save_params(self, episode: int):
