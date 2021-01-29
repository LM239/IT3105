from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.losses import Reduction
from configs.validate_configs import validate_critic_config
from keras.losses import MeanSquaredError
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class NeuralCritic:

    def __init__(self, critic_cfg, input_dim):
        validate_critic_config(critic_cfg)
        self.size = critic_cfg["hidden_layers"]

        self.lr = critic_cfg["lr"]
        self.eligibility_decay = critic_cfg["eligibility_decay"]
        self.discount_factor = critic_cfg["discount_factor"]

        self.model = Sequential()
        self.model.add(Input(shape=(input_dim,)))
        for layer in self.size:
            self.model.add(Dense(layer, activation='relu', use_bias=False))
        self.model.add(Dense(1, use_bias=False))
        self.model.compile(optimizer=SGD(learning_rate=self.lr), loss=MeanSquaredError(reduction=Reduction.NONE))

        self.eligibilities = [np.zeros(shape=(layer.input_shape[1], layer.output_shape[1])) for layer in
                              self.model.layers]
        self.episode = []

        self.deltas = []

    def update(self, episode, state, state_prime, reward):
        target = reward + self.discount_factor * self.model.predict([state_prime], batch_size=1)
        delta = target - self.model.predict([state], batch_size=1)
        self.episode.append((np.array([state]), target, delta))
        return delta

    def finish_episode(self):
        print("done")
        for state, target, delta in self.episode:
            self.fit([state], [target], delta)
        self.episode = []


    def reset_eligibilities(self):
        self.eligibilities = [np.zeros(shape=(layer.input_shape[1], layer.output_shape[1])) for layer in
                              self.model.layers]

    # Subclass this with something useful.
    def modify_gradients(self, gradients, delta):
        '''self.deltas.append(abs(delta[0][0]))
        if len(self.deltas) > 1200:
            plt.plot(self.deltas)
            plt.show()'''
        for index, el in enumerate(self.eligibilities):
            self.eligibilities[index] = self.eligibilities[index] * self.eligibility_decay * self.discount_factor + gradients[index] / (2*delta)
            gradients[index] = delta * self.eligibilities[index]
        return gradients

    # This returns a tensor of losses, OR the value of the averaged tensor.  Note: use .numpy() to get the
    # value of a tensor.
    def gen_loss(self, features, targets, avg=False):
        predictions = self.model(features)  # Feed-forward pass to produce outputs/predictions
        loss = self.model.loss(targets, predictions)  # model.loss = the loss function
        return tf.reduce_mean(loss).numpy() if avg else loss

    def fit(self, features, targets, delta, epochs=1):
        params = self.model.trainable_weights
        for epoch in range(epochs):
            with tf.GradientTape() as tape:  # Read up on tf.GradientTape !!
                loss = self.gen_loss(features, targets, avg=False)
                gradients = tape.gradient(loss, params)
                #print("Before:", np.shape(gradients))
                gradients = self.modify_gradients(gradients, delta)
                #print("After:", np.shape(gradients))
                #print("Zip: ", tuple(zip(gradients, params)))
                self.model.optimizer.apply_gradients(zip(gradients, params))

