
class Peg_solitaire:

    def __init__(self, config):
        if (config.type == "triangle"):
            self.state = [[1] * i for i in range(1, config.size + 1)]
        elif (config.type == "diamond"):
            self.state = [[1] * config.size] * config.size
        else:
            print("Unknown board type {}. Exiting".format(config.type))
            exit(1)
      
        return self
    
    
    def __str__(self):
        return "010101010"
    
    def vector(self):
        return [peg for peg in row for row in self.state]
    
