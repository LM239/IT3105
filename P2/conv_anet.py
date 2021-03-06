from tensorflow import float32

from configs.validate_configs import validate_anet_config
from interfaces.actornet import ActorNet
from keras.models import *
import tensorflow as tf
from keras.layers import *
import os

class ConvNet(ActorNet):
    def __init__(self, anet_cfg=None, board_size: int = 0, output_dim: int = 0, input_depth: int = 0, model_file: str = None):
        self.model = None
        self.file_type = ".h5"
        if anet_cfg is not None:
            validate_anet_config(anet_cfg)
            self.lr = anet_cfg["lr"]

            if "allow_cuda" in anet_cfg and not anet_cfg["allow_cuda"]:
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                if tf.test.gpu_device_name():
                    print('GPU found')
                else:
                    print("No GPU found")

            if "model_file" in anet_cfg:
                self.model = load_model(anet_cfg["model_file"])
                print("Loaded model from", anet_cfg["model_file"])
            else:
                self.input_boards = Input(shape=(board_size, board_size, input_depth))  # s: batch_size x board_x x board_y
                prev_layer = self.input_boards
                for depth, filter_size, dropout, padding in anet_cfg["cnn_filters"]:
                    prev_layer = Dropout(dropout)(Activation(anet_cfg["activation"])(BatchNormalization(axis=3)(Conv2D(depth, filter_size, padding=padding)(prev_layer))))  # batch_size  x board_x x board_y x num_channels
                prev_layer = Reshape((-1,))(prev_layer)
                dense_layers = anet_cfg["dense_layers"] if anet_cfg["dense_layers"] is not None else []
                for layer_size, dropout in dense_layers:
                    prev_layer = Dropout(dropout)(Activation(anet_cfg["activation"])(BatchNormalization(axis=1)(Dense(layer_size)(prev_layer))))
                self.pi = Dense(output_dim, activation='softmax', name='pi')(prev_layer)  # batch_size x self.action_size

                opt = type(tf.keras.optimizers.get(anet_cfg["optimizer"]))(learning_rate=anet_cfg["lr"])

                self.model = Model(inputs=self.input_boards, outputs=self.pi)
                self.model.compile(optimizer=opt, loss=anet_cfg["loss"])
            self.batch_size = anet_cfg["batch_size"]
        else:
            model_file = model_file if model_file.endswith(self.file_type) else model_file + self.file_type
            self.load_params(model_file)

    def train(self, features, targets):
        self.model.fit(
            tf.convert_to_tensor(features),  # training data
            tf.convert_to_tensor(targets),  # training targets
            epochs=1,
            batch_size=self.batch_size
        )

    def forward(self, input):
        input = tf.convert_to_tensor([input])
        return self.model(input, training=False)

    def save_params(self, path, file_name=None):
        print("-" * 20, "Saving anets to", path + file_name + self.file_type, "-" * 20)
        self.model.save(path + file_name + self.file_type)

    def load_params(self, model_file):
        self.model = load_model(model_file)


