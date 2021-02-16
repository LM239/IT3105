from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.losses import Reduction
from configs.validate_configs import validate_critic_config
from keras.losses import MeanSquaredError
import tensorflow as tf
import numpy as np
import os


class NeuralCritic:

    def __init__(self, critic_cfg, input_dim):

        # Retrieve values from config
        validate_critic_config(critic_cfg)
        self.size = critic_cfg["hidden_layers"]
        self.lr = critic_cfg["lr"]
        self.eligibility_decay = critic_cfg["eligibility_decay"]
        self.discount_factor = critic_cfg["discount_factor"]

        if "cuda" in critic_cfg and not critic_cfg["cuda"]:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            if tf.test.gpu_device_name():
                print('GPU found')
            else:
                print("No GPU found")

        # Build the network with first layer of input size, last layer with size 1,
        # and hidden layers with sizes from config
        self.model = Sequential()
        self.model.add(Input(shape=(input_dim,)))
        for layer in self.size:
            self.model.add(Dense(layer, activation='relu', use_bias=False))
        self.model.add(Dense(1, use_bias=False))
        self.model.compile(optimizer=SGD(learning_rate=self.lr), loss=MeanSquaredError(reduction=Reduction.NONE))

        # Eligibilities are initialised to zero for every weight
        self.eligibilities = []
        self.reset_eligibilities()

        # Array for remembering the last episode to use for network training
        self.episode = []

    def update(self, episode, state, state_prime, reward):
        target = reward + self.discount_factor * self.model.predict([state_prime], batch_size=1)
        delta = target - self.model.predict([state], batch_size=1)
        self.episode.append((np.array([state]), target, delta))
        return delta[0][0]

    def finish_episode(self):
        # Go through the episode and update the network with the observed values
        for state, target, delta in self.episode:
            self.fit([state], [target], delta)
        self.episode = []

    def reset_eligibilities(self):
        # Go through the shape of the network and set all eligibilities to zero for all weights
        self.eligibilities = [np.zeros(shape=(layer.input_shape[1], layer.output_shape[1])) for layer in
                              self.model.layers]

    def modify_gradients(self, gradients, delta):
        for index, el in enumerate(self.eligibilities):
            if not delta == 0.0:
                self.eligibilities[index] = self.eligibilities[index] * self.eligibility_decay * self.discount_factor + gradients[index] / (2*delta)
            gradients[index] = delta * self.eligibilities[index]
        return gradients

    def gen_loss(self, features, targets, avg=False):
        predictions = self.model(features)  # Feed-forward pass to produce outputs/predictions
        loss = self.model.loss(targets, predictions)  # model.loss = the loss function
        return tf.reduce_mean(loss).numpy() if avg else loss

    def fit(self, features, targets, delta, epochs=1):
        # Gets the gradients, modifies them, and then apply them
        params = self.model.trainable_weights
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                loss = self.gen_loss(features, targets, avg=False)
                gradients = tape.gradient(loss, params)
                gradients = self.modify_gradients(gradients, delta)
                self.model.optimizer.apply_gradients(zip(gradients, params))

