import numpy as np
import math
import warnings
import sys

from copy import deepcopy

class Gridworld:

    def __init__(self, height, width, sys_init, env_init, init_state_list):
        """
        A class that constructs the env on which both the layers learn to optimally perform the given task as per the
        reward function hand designed by a human operator. The env is of shape (N x M) where N is the number of rows
        and M is the  number of columns. Here we are playing on a square grid and hence N = M
        :param height:  N**2
        :type height: int
        :param width: M**2
        :type width: int
        :param sys_init: The initial cell in which the system starts : Pos (x)
        :type sys_init: int
        :param env_init: The initial cell in which the system starts : Pos (y)
        :type env_init: int
        :param init_state_list: a list of valid initial states
        :type init_state_list: list
        """
        self.height = height
        self.width = width
        # you always start in the state that belongs to the system player
        self.initPosition = (sys_init, env_init, 1)
        self.currentPosition = self.initPosition
        self.initStates = init_state_list

    def restart(self, sys_robot=None, env_robot=None):
        """
        A method to restart the env after each episode. If the initial positions of the robot are not given then choose
        a initial position randomly from the given lsit of valid initial states @initState
        :param sys_robot: The Pos value of the system robot
        :type sys_robot: int
        :param env_robot: The Pos value of the env robot
        :type env_robot: int
        :return: None
        :rtype: None
        """
        # only use this in case we dont specify an init state
        rand_init_state = np.random.choice(self.initStates)

        if sys_robot is not None:
            # the first tuple belongs to the system robot
            # since I cant directly manipulate a tuple I need to first convert into a list, modify it and then
            # change it back to a tuple
            lst = list(self.initPosition)
            lst[0] = sys_robot
            self.initPosition = tuple(lst)
        else:
            lst = list(self.initPosition)
            lst[0] = rand_init_state.x
            self.initPosition = tuple(lst)

            # self.initPosition[0] = sys_robot
        if env_robot is not None:
            # the second tuple belongs to the env robot
            # self.initPosition[1] = env_robot
            lst = list(self.initPosition)
            lst[1] = env_robot
            self.initPosition = tuple(lst)
        else:
            lst = list(self.initPosition)
            lst[1] = rand_init_state.y
            self.initPosition = tuple(lst)

        # TODO check if this initializes in the format (x, y, 1). The 'one' should be present in the tuple
        self.currentPosition = deepcopy(self.initPosition)

    def play_one_player_at_a_time(self, player, action):
        self.move(player, action)

        return self._get_reward(self.currentPosition)

    def play(self, sys_action, env_action):
        """
        A method to get the next state on the gridworld given an action for the  system and the env robot. This method
        calls move method to evolve for each player respectively.
        :param sys_action: A valid action for the system robot
        :type sys_action: string
        :param env_action: A valid action for the env robot
        :type env_action: string
        :return: reward a binary value (+1 or 0) +1 if you both the robot are in diagonally opposite position else 0
        :rtype: int
        """
        # evolve according to system player and then env player. Note, the sequence does not matter as the both the
        # actions are taken simultaneously and the updated cell position is used in the code.
        # a fancy way to just randomly choose a player and first evolve him
        player = np.random.randint(0, 2)
        actions = [sys_action, env_action]
        self.move(player, actions[player])
        self.move(1-player, actions[1 - player])

        return self._get_reward(self.currentPosition)

    def move(self, player, action):
        """
        A method to evolve a player's position as per the action. As the currentPosition hold the system robot first and
        then the env player, system player will be referred as 0 and env robot will be reffered as 2. This method updates
        the current position on the gridworld.
        :param player: the value of the player. 0 - system player/robot and 1 - env player/robot
        :type player: int
        :param action: A valid action that belongs to the action set (A_c U A_uc)
        :type action: string
        :return:
        :rtype:
        """
        newPosition = self.currentPosition[player] + self.actionToMove(action)
        # since tuples don't support item assignment, converting it to list, manipulating it and then converting it
        # back to a tuple
        lst = list(self.currentPosition)
        lst[player] = newPosition
        self.currentPosition = tuple(lst)

    def actionToMove(self, action):
        # A_c : [up_s, down_s, left_s, right_s, stay_s]
        # A_uc: [up_e, down_e, left_e, right_e]
        switch_case = {
            "up_s": +4,
            "down_s": -4,
            "left_s": -1,
            "right_s": +1,
            "stay_s": 0,
            "up_e": +4,
            "down_e": -4,
            "left_e": -1,
            "right_e": +1,
        }

        return switch_case.get(action, "Not a valid action")

    def _convert_pos_to_xy(self, location):
        """
        A method to convert location of format (pos x pos x player) to (x1,y1) , (x2,y2)
        :param location: a tuple of type (pos x pos x player)
        :type location: tuple
        :return: A tuple containing the system position and env position
        :rtype: tuple
        """
        sys_pos = location[0]  # an integer value of a cell
        env_pos = location[1]  # an integer value of a cell

        # initialize sys_xy and env_xy
        sys_xy = None
        env_xy = None
        # assuming that we are in a gridworld n = m and max(cell_value) = n**2 -1
        # get original env width(n) and height(m)
        org_n = int(math.sqrt(self.height))
        # since its a square world
        org_m = org_n
        # convert pos to x,y
        # finding a value of x and y that satifies pos = n*y + x
        for x in range(org_n):
            for y in range(org_m):
                if org_n * y + x == sys_pos:
                    sys_xy = (x, y)
                if org_n * y + x == env_pos:
                    env_xy = (x, y)

        if isinstance(sys_xy, type(None)) or isinstance(env_xy, type(None)):
            warnings.warn("Could not convert the pos value of system or env robot to its respective (x,y) values. " \
                          "Ideally this should have never occurred. The system pos value is", sys_pos, "and the env " \
                          "pos values is", env_pos, ". Exiting code")
            sys.exit(-1)
        return sys_xy, env_xy

    def _get_reward(self, new_location):
        """
        A method to compute the reward. The reward function returns +1 if both the robots in the diagonally opposite
        position else 0
        :param new_location: a tuple of the form (x, y, 1) : x - Pos value of the system robot anf y - Pos value to the
        env robot.
        :type new_location: tuple
        :return:  return +1 if the agents lie diagonally opposite else 0
        :rtype: int
        """

        # convert the Pos state value to (x,y) coordinates
        sys_xy, env_xy = self._convert_pos_to_xy(new_location)
        # system xy coordinates
        x1 = sys_xy[0]
        y1 = sys_xy[1]
        # env xy coordinates
        x2 = env_xy[0]
        y2 = env_xy[1]
        # two points X, Y are diagonally opposite if abs(x1 - x2) = abs(y1 - y2)
        if abs(x1 - x2) == abs(y1 - y2):
            return +1
        else:
            return 0


if __name__ == '__main__':
    print("******************Class Gridworld main file******************")