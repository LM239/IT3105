from configs.validate_configs import validate_anet_config
from interfaces.actornet import ActorNet
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import *
import tensorflow as tf
from keras.layers import *

class Anet(ActorNet):
    def __init__(self, anet_cfg=None, board_size: int = 0, output_dim: int = 0, model_file: str = None):
        if model_file is None:
            validate_anet_config(anet_cfg)
            self.size = anet_cfg["hidden_layers"]
            self.lr = anet_cfg["lr"]

            self.file_structure = anet_cfg["file_structure"]

            # Build the network with first layer of input size
            # and hidden layers with sizes from config

            #from https://github.com/suragnair/alpha-zero-general/blob/master/gobang/keras/GobangNNet.py
            self.input_boards = Input(shape=(board_size, board_size))  # s: batch_size x board_x x board_y

            x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)  # batch_size  x board_x x board_y x 1
            h_conv1 = Activation(anet_cfg["activation"])(BatchNormalization(axis=3)(Conv2D(anet_cfg["l1_filters"], 3, padding='same')(x_image)))  # batch_size  x board_x x board_y x num_channels
            h_conv2 = Activation(anet_cfg["activation"])(BatchNormalization(axis=3)(Conv2D(anet_cfg["l2_filters"], 3, padding='same')(h_conv1)))  # batch_size  x board_x x board_y x num_channels
            h_conv3 = Activation(anet_cfg["activation"])(BatchNormalization(axis=3)(Conv2D(anet_cfg["l3_filters"], 3, padding='valid')(h_conv2)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
            h_conv3_flat = Flatten()(h_conv3)
            s_fc1 = Dropout(anet_cfg["dropout"])(Activation(anet_cfg["activation"])(BatchNormalization(axis=1)(Dense(1024)(h_conv3_flat))))  # batch_size x 1024
            s_fc2 = Dropout(anet_cfg["dropout"])(Activation(anet_cfg["activation"])(BatchNormalization(axis=1)(Dense(512)(s_fc1))))  # batch_size x 1024
            self.pi = Dense(output_dim, activation='softmax', name='pi')(s_fc2)  # batch_size x self.action_size
            #self.v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1

            opt = type(tf.keras.optimizers.get(anet_cfg["optimizer"]))(learning_rate=anet_cfg["lr"])

            self.model = Model(inputs=self.input_boards, outputs=[self.pi]) #, self.v])
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

