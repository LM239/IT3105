
class PegSolitaire:

    def __init__(self, config):
        if config.type == "triangle":
            self.state = [[1] * i for i in range(1, config.size + 1)]
        elif config.type == "diamond":
            self.state = [[1] * config.size] * config.size
        else:
            print("Unknown board type {}. Exiting".format(config.type))
            exit(1)
    
    def __str__(self):
        val = ""
        return val.join(str(peg) for row in self.state for peg in row)
    
    def vector(self):
        return [peg for row in self.state for peg in row]

