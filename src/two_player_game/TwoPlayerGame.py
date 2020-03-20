# file to construct the game and extract admissible strategies
from src.two_player_game.GridWorld import GridWorld
import sys
import random

random.seed(1)


class TwoPlayerGame:
    """
    Summary of the class

    This class constructs a two-player game graph (G) on which a set of winning/permissive strategies W will
    computed that are realizable on the G.

    Attributes:
        env : A instance of the environment that initializes an empty set of states
        and sets the Pos variable. Pos = {0, ...., N^2 - 1} set of cells in the map
        S : Set of states (Pos x Pos X {0,1})
        S_s : Set of states that belong to the system (Pos x Pos x {1})
        S_e : Set of states that belong to the environment (Pos x Pos x {0})
        A_c : finite set of control actions of the system
        A_uc : finite set of uncontrolled actions of the environment
        T : Transition function (T : S x {A_c U A_uc} -> 2^S)
        W : Winning condition
    """

    def __init__(self):
        """
        Construct a two player game. Type "help(TwoPlayerGame)" to knwo more about the attributes of this class

        attributes :

        confg : a dictionary that maps every type of variable to its config file {{env : env_config},
        {player: player_config},...}
        """
        self.conf = {}
        self.env = None
        self.S = None
        self.S_s = None
        self.S_e = None
        self.A_c = None
        self.A_uc = None
        self.Init = None
        self.T = None
        self.W = None


    def print_env_states(self):
        """
        pass instance of gridworld
        :param gridworld:
        :type gridworld: GridWorld
        :return: None

        """

        for s_r in self.S:
            for s_c in s_r:
                print("state : ({},{},{})".format(s_c.x, s_c.y, s_c.t))
        # print(list(gridworld.states))


    def initialize_states(self):
         """
         Method to initialize a set of states with initially no state assigned a player
         :return:
         :rtype:
         """

         self.S = self.env.create_states()

    def set_state_players(self):
        for row_state in self.S:
            for col_state in row_state:
                # GridWorld.set_state_player(col_state, random.randint(0, 1), self.env)
                self.env.set_state_player(col_state, random.randint(0, 1))

    def get_state_player(self, location):
        """
        get the player assigned to each state
        :param location: x and y position of the state
        :type location: tuple (x,y)
        :return: state t (s.t) value
        :rtype: int {0, 1} env_player = 0 system_player = 1
        """
        return self.S[location[0]][location[1]].t

    def create_env(self):
        """
        Method yo create an instance of the environment which holds the range of the cells and an
        empty list of states is initialized.
        :return:
        :rtype:
        """
        # TODO looks like unnecessary intialization; look at saving space and memory
        config_dictionay = self.get_config_filename()

        # get env config
        env_confg = config_dictionay.get("env")

        # create instance of the environment
        self.env = GridWorld(env_confg)
        return self.env

    # get env config name
    def get_config_filename(self):
        return self.conf

    def add_config_filename(self, key, value):
        return self.conf.update({key: value})

def main():
    # load env paramter from a config file
    yaml_name = "env"
    config_file_name = (str(sys.path[0]) + "/config/" + yaml_name)

    # create an instance of the two player game
    game = TwoPlayerGame()
    game.add_config_filename("env", config_file_name)

    # create env
    game.create_env()

    # initialize states
    game.initialize_states()

    # set fake player states
    # TODO replace this in future with some useful feature
    game.set_state_players()

    # method to print all the states in the environment
    game.print_env_states()




if __name__ == "__main__":

   main()
