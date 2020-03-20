# file to construct the game and extract admissible strategies
from src.two_player_game.GridWorld import GridWorld
import sys
import random
import yaml
import numpy as np

random.seed(1)

def create_yaml_path_name(yaml_file_name):
    """
    helper class to create paths that can then be directly used in the TwoPlayerGame class
    :return:
    :rtype: path
    """
    file_path = (str(sys.path[0]) + "/config/" + yaml_file_name)
    return file_path


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
        Note : We assume G to be deterministic i.e | T(s, a) | for all (s that belong to S) and (a that belong to A(s))
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

    @staticmethod
    def read_yaml_file(configFiles):
        """
            reads the yaml file data and stores them in appropriate variables
        """
        with open(configFiles + ".yaml", 'r') as stream:
            data_loaded = yaml.safe_load(stream)

        return data_loaded


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

    def get_players_data(self):
        # converting the values in the controlled action and states into tuples
        players = TwoPlayerGame.read_yaml_file(self.conf.get("player"))
        players["system_player"] = {k: tuple(v) for k, v in players["system_player"].items()}
        players["env_player"] = {k: tuple(v) for k, v in players["env_player"].items()}
        return players

    def set_controlled_actions(self):
        players = self.get_players_data()
        # get the controlled actions tuple from players
        self.A_c = players["system_player"].get("A_c")

    def get_controlled_actions(self):
        return self.A_c

    def set_uncontrolled_actions(self):
        player = self.get_players_data()
        self.A_uc = player["env_player"].get("A_uc")

    def set_state_actions(self, state):
        """
        method to assign a set of actions available at a given state
        :param state:
        :type state:
        :return: None
        """
        state.action = self.A_c if state.get_state_player(state) else self.A_uc

    def get_state_actions(self, state):
        return state.action


    def get_uncontrolled_actions(self):
        return self.A_uc

    def get_controlled_action(self):
        return self.A_c

    def set_initial_state(self):

        # get a random state and assing it to the system where x != y
        init_state = random.choice(random.choice(self.S))
        while init_state.x == init_state.y:
            init_state = random.choice(self.S)

        init_state.set_state_player(init_state, player=1)
        self.Init = init_state

    def get_initial_state(self):
        return self.Init

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

    # create yaml paths
    env_yaml = "env"
    player_yaml = "player"

    # create an instance of the two player game
    game = TwoPlayerGame()
    game.add_config_filename("env", create_yaml_path_name(env_yaml))
    game.add_config_filename("player", create_yaml_path_name(player_yaml))

    # create env
    game.create_env()

    # initialize states
    game.initialize_states()

    # set fake player states
    # TODO replace this in future with some useful feature
    # game.set_state_players()

    # method to print all the states in the environment
    game.print_env_states()

    # set controlled and uncontrolled action tuples
    game.set_controlled_actions()
    game.set_uncontrolled_actions()

    # set init state
    game.set_initial_state()
    game.set_state_actions(game.Init)
    # get controlled and uncontrolled action
    print(game.get_initial_state().x,game.get_initial_state().y, game.get_initial_state().t)
    print(game.get_state_actions(game.Init))

if __name__ == "__main__":
   main()
