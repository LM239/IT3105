from BasicClientActorAbs import BasicClientActorAbs
from players import GreedyPlayer, ProbabilisticPlayer, Player
from conv_anet import ConvNet
from worlds.hex_world import HexWorld
import os
import yaml
import getopt
import sys

class BasicClientActor(BasicClientActorAbs):

    def __init__(self, player: Player, IP_address=None, verbose=True):
        self.series_id = -1
        BasicClientActorAbs.__init__(self, IP_address, verbose=verbose)



    def handle_get_action(self, state):
        """
        Here you will use the neural net that you trained using MCTS to select a move for your actor on the current board.
        Remember to use the correct player_number for YOUR actor! The default action is to select a random empty cell
        on the board. This should be modified.
        :param state: The current board in the form (1 or 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), where
        1 or 2 indicates the number of the current player.  If you are player 2 in the current series, for example,
        then you will see a 2 here throughout the entire series, whereas player 1 will see a 1.
        :return: Your actor's selected action as a tuple (row, column)
        """

        # This is an example player who picks random moves. REMOVE THIS WHEN YOU ADD YOUR OWN CODE !!
        state_active = len([1 for i in state[1:] if i != 0])
        me = (1, 0) if state_active % 2 == 0 else (0, 1)
        op = tuple(reversed(me))
        state = [me if cell == self.series_id else (op if cell > 0 else (0, 0)) for cell in state[1:]] + [me]
        action = player.play(state)
        next_move = divmod(action, 6)
        #############################
        #
        #
        # YOUR CODE HERE
        #
        # next_move = ???
        ##############################
        return next_move

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """
        Set the player_number of our actor, so that we can tell our MCTS which actor we are.
        :param unique_id - integer identifier for the player within the whole tournament database
        :param series_id - (1 or 2) indicating which player this will be for the ENTIRE series
        :param player_map - a list of tuples: (unique-id series-id) for all players in a series
        :param num_games - number of games to be played in the series
        :param game_params - important game parameters.  For Hex = list with one item = board size (e.g. 5)
        :return

        """
        self.series_id = series_id
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_start(self, start_player):
        """
        :param start_player: The starting player number (1 or 2) for this particular game.
        :return
        """
        self.starting_player = start_player
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_over(self, winner, end_state):
        """
        Here you can decide how to handle what happens when a game finishes. The default action is to print the winner and
        the end state.
        :param winner: Winner ID (1 or 2)
        :param end_state: Final state of the board.
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        ##############################
        print("Game over, these are the stats:")
        print('Winner: ' + str(winner == self.series_id))
        print('End state: ' + str(end_state))

    def handle_series_over(self, stats):
        """
        Here you can handle the series end in any way you want; the initial handling just prints the stats.
        :param stats: The actor statistics for a series = list of tuples [(unique_id, series_id, wins, losses)...]
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Series ended, these are the stats:")
        print(str(stats))
        total_wins = 0
        total_losses = 0
        for unique_id, series_id, wins, losses in stats:
            total_wins += wins
            total_losses += losses

    def handle_tournament_over(self, score):
        """
        Here you can decide to do something when a tournament ends. The default action is to print the received score.
        :param score: The actor score for the tournament
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Tournament over. Your score was: " + str(score))

    def handle_illegal_action(self, state, illegal_action):
        """
        Here you can handle what happens if you get an illegal action message. The default is to print the state and the
        illegal action.
        :param state: The state
        :param action: The illegal action
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("An illegal action was attempted:")
        print('State: ' + str(state))
        print('Action: ' + str(illegal_action))


if __name__ == '__main__':
    short_options = "h"
    long_options = ["help"]  # command line options use either python run.py -h | --help

    try:
        arguments, values = getopt.getopt(sys.argv[1:], short_options, long_options)  # set up cmd options
        for current_argument, current_value in arguments:  # check options
            if current_argument in ("-h", "--help"):
                print("Usage: python {} <path-to-config>".format(sys.argv[0]))  # print help
                exit(0)
        file_name = sys.argv[1] if len(sys.argv) > 1 else "configs/oht.yaml"
        with open(file_name) as file:  # try to open config file
            oht_config = yaml.load(file, Loader=yaml.FullLoader)
            file.close()
    except getopt.error as err:
        print(str(err))
        sys.exit(2)
    except FileNotFoundError:
        print("Could not find file at {}\nExiting".format(sys.argv[1]))
        exit(1)

    if "oht" not in oht_config:
        print("Missing required config dict 'oht' \nExiting")
        exit(1)

    if "sim_world" not in oht_config:
        print("Missing required config dict 'sim_world' \nExiting")
        exit(1)

    if "player_type" not in oht_config["oht"]:
        print("Missing required config value 'player_type' \nExiting")
        exit(1)

    if oht_config["oht"]["player_type"] not in ["probabilistic", "greedy"]:
        print("Unknown player type {}, valid types are: {} \nExiting".format(oht_config["player_type"],
                                                                             ["probabilistic", "greedy"]))
        exit(1)

    world_config = oht_config["sim_world"]

    bca_cfg = oht_config["oht"]
    world_manager = None

    if "anet_file" not in bca_cfg:
        print("Missing required config value 'player_type' \nExiting")
        exit(1)

    if world_config["world"] == "hex":  # create sim_world for the actor critic
        world_manager = HexWorld(world_config, 0)
    else:
        print("Unknown world type: {} \n Exiting".format(world_config["world"]))
        exit(1)

    if "allow_cuda" in oht_config and not oht_config["allow_cuda"]:
        print("Disable cuda")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    anet = ConvNet(model_file=bca_cfg["anet_file"])
    player = ProbabilisticPlayer(anet, world_manager, "me") if oht_config["oht"]["player_type"] == "probabilistic" else GreedyPlayer(anet, world_manager, "me")
    bsa = BasicClientActor(player, verbose=False)
    bsa.connect_to_server()
