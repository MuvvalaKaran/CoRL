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
        TODO: IS state is a dictionary or list. Look at which implementation is useful
        :param filename:
        :type filename:
        """
        GridWorld.clear_shell_screen()
        self.filename = filename
        env_data = GridWorld.read_yaml_file(filename)
        self.pos_x = range(0, env_data["env_size"]["n"]**2, 1)
        self.pos_y = range(0, env_data["env_size"]["m"]**2, 1)
        self.states = []
        # grifworld will have a collection of states


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
        method to create state for the gridworld
        """
        for ix in self.pos_x:
            for iy in self.pos_y:
                # create states
                s = State(self.filename)
                s.setstate(ix, iy, player=player)
                self.states.append(s)

    #TODO write get and set state methods too
    def set_state(self, player):

        if player is None:
            print("Please specify player value")
            raise ValueError



class State(GridWorld):

    def __init__(self, filename):
        """
        state inherits from gridworld class has the follwoing attributes
        x_pos = x position of state (0, n^2 -1)
        y_pos  = y position of state (0, m^2 - 1)
        t_value = which player does the state belong to
        t_value = 0 - system
        t_value = 1 - environment
        """
        super().__init__(filename)
        self.x = None
        self.y = None
        self.t = None

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
        if x >= self.pos_x.stop or x < 0:
            print("Please make sure x is non-negative and less than {}", format(self.pos_x.stop - 1))
            raise ValueError

        if y >= self.pos_y.stop or y < 0:
            print("Please make sure x is non-negative and less than {}", format(self.pos_x.stop - 1))
            raise ValueError

        if player is None:
            message = "Player is set to None. You should assign a state to a player"
            warnings.warn(message)

        self.x = x
        self.y = y
        self.t = player

    # is this method required?
    def getstate(self):
        return self