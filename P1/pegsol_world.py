
class Peg_solitaire:

    def __init__(self, config):
        if (config.type == "triangle"):
            self.state = [[1] * i for i in range(1, config.size + 1)]
        elif (config.type == "diamond"):
            self.state = [[1] * config.size] * config.size
        else:
            print("Unknown board type {}. Exiting".format(config.type))
            exit(1)
      
        return 0
