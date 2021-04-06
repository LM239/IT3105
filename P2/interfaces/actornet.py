# Interface for neural network
class ActorNet():
    # Updates network
    def train(self, features, target):
        pass

    # Makes prediction from input
    def forward(self, input):
        pass

    # Saves current network params to file
    def save_params(self, path, file_name):
        pass

    # Loads network from file
    def load_params(self, model_file):
        pass
