from keras.optimizers import SGD
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
        model.compile(optimizer=SGD(learning_rate=self.lr), loss='mse')

        self.split_gd = SplitGD(model)
        self.episode = []

    def update(self, episode, state, state_prime, reward):
        V_star = reward + self.discount_factor * self.split_gd.model.predict(state_prime)
        self.episode.append((state, V_star))

    def finish_episode(self):
        for state, target in self.episode:
            self.split_gd.fit([state], [target], vfrac=0.0, verbosity=0)
        self.episode = []

    def reset_eligibilities(self):
        raise NotImplementedError()
