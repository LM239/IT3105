import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from splitgd import SplitGD
from configs.validate_configs import validate_critic_config

class NeuralCritic:

    def __init__(self, critic_cfg, input_dim):
        validate_critic_config(critic_cfg)
        self.size = critic_cfg["hidden_layers"]

        self.lr = critic_cfg["lr"]
        self.eligibility_decay = critic_cfg["eligibility_decay"]
        self.discount_factor = critic_cfg["discount_factor"]

        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        for layer in self.size:
            model.add(Dense(layer, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='sgd', loss='mse')

        self.split_gd = SplitGD(model)

    def update(self, episode, state, state_prime, reward):
        target = reward + self.discount_factor * self.split_gd.model.predict(state_prime)
        self

    def reset_eligibilities(self):
        raise NotImplementedError()
