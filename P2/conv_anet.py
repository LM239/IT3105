from tensorflow import float32

from configs.validate_configs import validate_anet_config
from interfaces.actornet import ActorNet
from keras.models import *
import tensorflow as tf
from keras.layers import *
import os

class ConvNet(ActorNet):
    def __init__(self, anet_cfg=None, board_size: int = 0, output_dim: int = 0, model_file: str = None):
        self.model = None
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
                self.input_boards = Input(shape=(board_size, board_size, 1))  # s: batch_size x board_x x board_y
                prev_layer = self.input_boards
                for depth, dropout, padding in anet_cfg["cnn_filters"]:
                    prev_layer = Dropout(dropout)(Activation(anet_cfg["activation"])(BatchNormalization(axis=3)(Conv2D(depth, 3, padding=padding)(prev_layer))))  # batch_size  x board_x x board_y x num_channels
                prev_layer = Reshape((-1,))(prev_layer)

                for layer_size, dropout in anet_cfg["dense_layers"]:
                    prev_layer = Dropout(dropout)(Activation(anet_cfg["activation"])(BatchNormalization(axis=1)(Dense(layer_size)(prev_layer))))
                self.pi = Dense(output_dim, activation='softmax', name='pi')(prev_layer)  # batch_size x self.action_size

                opt = type(tf.keras.optimizers.get(anet_cfg["optimizer"]))(learning_rate=anet_cfg["lr"])

                self.model = Model(inputs=self.input_boards, outputs=self.pi)
                self.model.compile(optimizer=opt, loss=anet_cfg["loss"])
            self.batch_size = anet_cfg["batch_size"]
        else:
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
        print("-" * 20, "Saving anets to", path + file_name, "-" * 20)
        self.model.save(path + file_name)

    def load_params(self, model_file):
        self.model = load_model(model_file)


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class ConvNet(nn.Module):
    def __init__(self, anet_cfg=None, board_size: int = 0, output_dim: int = 0, model_file: str = None):
        if model_file is None:
            # game params
            super(ConvNet, self).__init__()
            self.board_x, self.board_y = board_size, board_size
            self.conv1 = nn.Conv2d(1, anet_cfg["l1_filters"], 3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(anet_cfg["l1_filters"], anet_cfg["l2_filters"], 3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(anet_cfg["l2_filters"], anet_cfg["l3_filters"], 3, stride=1)

            self.bn1 = nn.BatchNorm2d(anet_cfg["l1_filters"])
            self.bn2 = nn.BatchNorm2d(anet_cfg["l2_filters"])
            self.bn3 = nn.BatchNorm2d(anet_cfg["l3_filters"])

            self.fc1 = nn.Linear(anet_cfg["l3_filters"] * (self.board_x-2)*(self.board_y-2), 1024)
            self.fc_bn1 = nn.BatchNorm1d(1024)

            self.fc2 = nn.Linear(1024, 512)
            self.fc_bn2 = nn.BatchNorm1d(512)

            self.fc3 = nn.Linear(512, output_dim)

            self.fc4 = nn.Linear(512, 1)
            self.dropout = anet_cfg["dropout"]
            self.l3_filters = anet_cfg["l1_filters"]
            self.file_structure = anet_cfg["file_structure"]
        else:
            #self.model = load_model(model_file)
            pass

    def forward(self, s):
        with torch.no_grad():
            return self.forward_grad(s)

    def forward_grad(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = torch.tensor(s, dtype=torch.float32)
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))
        s = torch.flatten(s, 1)

        s = self.fc1(s)
        if s.shape[0] > 1:
            s = self.fc_bn1(s)
        s = F.dropout(F.relu(s), p=self.dropout, training=self.training)  # batch_size x 1024
        s = self.fc2(s)
        if s.shape[0] > 1:
            s = self.fc_bn2(s)
        s = F.dropout(F.relu(s), p=self.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)
        return F.softmax(pi, dim=1).numpy()

    def save_params(self, episode: int):
        file_name = self.file_structure + "checkpoint_" + str(episode) + ".h5"
        torch.save(self, file_name)

"""

