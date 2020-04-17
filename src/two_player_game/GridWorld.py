# class to construct the gridworld and return the states
import yaml
import warnings

from os import system as system_call
from platform import system as system_name


class GridWorld(object):

    def __init__(self, filename):
        """

        Create states in the environment
        pos : {0, N^2 -1}
        State is tuple S = {pos x pos x {0,1}}
        TODO: IS state a dictionary or list. Look at which implementation is useful
        :param filename:
        :type filename:
        """
        # grifdorld will have a collection of states
        GridWorld.clear_shell_screen()
        self.filename = filename
        self.env_data = GridWorld.read_yaml_file(filename)
        self.pos_x = range(0, self.env_data["env_size"]["n"]**2, 1)
        self.pos_y = range(0, self.env_data["env_size"]["m"]**2, 1)
        self.states = []

    @staticmethod
    def clear_shell_screen():
        """
        clears the shell screen
        :return: None
        :rtype: None
        """
        cmd = "cls" if system_name().lower() == "windows" else "clear"
        system_call(cmd)

    @staticmethod
    def read_yaml_file(filename):
        """
        reads the yaml and build a grid world according
        :return:
        :rtype:
        """
        with open(filename + ".yaml", "r") as stream:
            data = yaml.safe_load(stream)
        return data

    def create_states(self, player=None):
        """
        MEthod to initialize states
        method to create state for the gridworld
        """
        # for every row
        for ix in self.pos_x:
            # for every column
            col_list = []
            for iy in self.pos_y:
                # for the range of players
                player_list = []
                for p in range(player):
                    # create states and add them to the dictionary of states
                    s = State()
                    # assuming a state with (x,y) exists
                    s.setstate(ix, iy, p)
                    # self.states.update(s.state)
                    player_list.append(s)
                col_list.append(player_list)
            # a list of lists
            self.states.append(col_list)

        return self.states

    # @staticmethod
    def set_state_player(self, state, player):
        """
        Function to assign a state to each player
        :param state: state of the gridworld
        :type state: instance of State
        :param player:
        :type player:
        :return:
        :rtype:
        """
        if player is None:
            message = "Player is set to None. You should assign a state to a player"
            warnings.warn(message)

        state = self.get_state(state.x, state.y)
        state.t = player

        # return (s.x, s.y), s.t
    # @staticmethod
    def get_state(self, x, y):
        """
        Function to get a state
        :param x:
        :type x:
        :param y:
        :type y:
        :return:
        :rtype: State instance
        """

        # get the state with (x, y)
        # if gm.states.get((x, y)) is None:
        #     print("State with ({},{}) position does not exist".format(x, y))
        #     return
        return self.states[x][y]


class State:

    def __init__(self):
        """
        state inherits from gridworld class has the follwoing attributes
        x_pos = x position of state (0, n^2 -1)
        y_pos  = y position of state (0, m^2 - 1)
        t_value = which player does the state belong to
        t_value = 0 - system
        t_value = 1 - environment
        actions = actions avaiable to a state (depends on owner of the state)
        """
        # super().__init__(filename)
        self.state = {}
        self.x = None
        self.y = None
        self.t = None
        self.action = None

    def setstate(self, x=None, y=None, player=None):
        """

        :param x: {0 , n^2 - 1}
        :type x: int
        :param y: {0, m^2 - 1}
        :type y: int
        :param player:
        :type player: int {0,1}
        :return: None
        """
        # if x >= self.pos_x.stop or x < 0:
        #     print("Please make sure x is non-negative and less than {}", format(self.pos_x.stop - 1))
        #     raise ValueError
        #
        # if y >= self.pos_y.stop or y < 0:
        #     print("Please make sure x is non-negative and less than {}", format(self.pos_x.stop - 1))
        #     raise ValueError

        self.x = x
        self.y = y
        self.t = player

        self.state = {(self.x, self.y): self.t}

    # is this method required?
    def getstate(self):
        return self

    def set_state_player(self, state, player):
        """
           Function to assign a state to each player
           :param state: state of the gridworld
           :type state: instance of State
           :param player:
           :type player:
           :return:
           :rtype:
           """
        if player is None:
            message = "Player is set to None. You should assign a state to a player"
            warnings.warn(message)

        # state = self.get_state(state.x, state.y)
        state.t = player
        state.state = {(self.x, self.y): self.t}

    def get_state_player(self, state):
        return state.t
