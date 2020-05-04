# file to construct the game and extract admissible strategies
from two_player_game.GridWorld import GridWorld
import sys
import random
import yaml
import re
import os
import numpy as np

from itertools import product

# seed randomness for testing
random.seed(1)
prompt_user = False

# print the env states constructed
print_env_states = False
# set random state players flag to test features
set_fake_state_player = False

def get_cwd_path():
    return os.path.dirname(os.path.realpath(__file__))

def create_yaml_path_name(yaml_file_name):
    """
    helper class to create paths that can then be directly used in the TwoPlayerGame class
    :return:
    :rtype: path
    """
    file_path = (get_cwd_path() + "/config/" + yaml_file_name)
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
        Ap ; Set of atomic proposiitons
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
        self.S_s = []
        self.S_e = []
        self.A_c = None
        self.A_uc = None
        self.Init = None
        self.T = None
        self.W = {"labeling_func": self.labeling_function}
        self.AP = None

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
                for s_t in s_c:
                    print(f"state : ({s_t.x},{s_t.y},{s_t.t})")

    def initialize_states(self):
         """
         Method to initialize a set of states with initially no state assigned a player
         :return:
         :rtype:
         """
         # for i in range(0, 2):
         self.S = self.env.create_states(2)

    def set_state_players(self):
        for row_state in self.S:
            for col_state in row_state:
                for state in col_state:
                    self.env.set_state_player(state, random.randint(0, 1))

    def get_state_player(self, state, location=None):
        """
        get the player assigned to each state
        :param location: x and y position of the state
        :type location: tuple (x,y)
        :return: state t (s.t) value
        :rtype: int {0, 1} env_player = 0 system_player = 1
        """

        return state.t

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
        # A_c : [up_s, down_s, left_s, right_s, stay_s]
        # A_uc: [up_e, down_e, left_e, right_e]
        up_action = re.compile('^up')
        down_action = re.compile('^down')
        left_action = re.compile('^left')
        right_action = re.compile('^right')

        state.action = self.A_c if state.get_state_player(state) else self.A_uc
        if state.x == 0:
            # no left action is available
            if any(left_action.match(act) for act in state.action):
                # get the exact literal name and remove it
                act = [act for act in state.action if left_action.match(act)]
                state.action = list(state.action)
                state.action.remove(act[0])
                state.action = tuple(state.action)

        if state.x == self.env.env_data["env_size"]["m"]**2 - 1:
            # no right action is available
            if any(right_action.match(act) for act in state.action):
                # get the exact literal name and remove it
                act = [act for act in state.action if right_action.match(act)]
                state.action = list(state.action)
                state.action.remove(act[0])
                state.action = tuple(state.action)

        if state.y == 0:
            # no down action
            if any(down_action.match(act) for act in state.action):
                # get the exact literal name and remove it
                act = [act for act in state.action if down_action.match(act)]
                state.action = list(state.action)
                state.action.remove(act[0])
                state.action = tuple(state.action)

        if state.y == self.env.env_data["env_size"]["n"]**2 - 1:
            # no up action
            if any(up_action.match(act) for act in state.action):
                # get the exact literal name and remove it
                act = [act for act in state.action if up_action.match(act)]
                state.action = list(state.action)
                state.action.remove(act[0])
                state.action = tuple(state.action)

    def get_state_actions(self, state):
        return state.action

    def get_uncontrolled_actions(self):
        return self.A_uc

    # def get_controlled_action(self):
    #     return self.A_c

    def set_initial_state(self):
        # all states that belong to the system player are all by default the initial states which will later be pruned
        # from the strategy we get from slugs
        self.Init = self.get_sys_states()

    def get_initial_state(self):
        return self.Init

    def _initialize_states(self):
        # method to assign each player its respective states
        for row_state in self.S:
            for col_state in row_state:
                for state in col_state:
                    if state.t == 1:
                        self._set_sys_state(state)
                    else:
                        self._set_env_state(state)

    def _set_sys_state(self, state):
        self.S_s.append(state)

    def get_sys_states(self):
        return self.S_s

    def _set_env_state(self, state):
        self.S_e.append(state)

    def get_env_states(self):
        return self.S_e

    def create_env(self):
        """
        Method to create an instance of the environment which holds the range of the cells and an
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

    def tranisition_function(self, state, action):
        """
        A mapping from a state and action to the next state
        Note : Our game is deterministic i.e |T(.)| <= 1
        The controlled actions only manipulate the x values while
        the uncontrolled actions only manipulate the y values
        :return: Nest state
        :rtype: state
        """

        # create a dictionary of mapping for the system and the player
        # act_state_dct = {
        #     "up" : "+4",
        #     "down" : "-4",
        #     "left" : "+1",
        #     "right" : "-1"
        # }

        # TODO add regex expression to make this more robust in future

        # get the x and y values of the state
        state_x = state.x
        state_y = state.y
        # check which player the current state belongs and according change the x/y values of the state
        # if the state belongs to the env

        if state.t == 0:
            if action == "left_e":
                state_y -= 1
            elif action == "right_e":
                state_y += 1
            elif action == "up_e":
                state_y += 4
            else:
                state_y -= 4

        # if the state belongs to the system
        elif state.t == 1:
            if action == "left_s":
                state_x -= 1
            elif action == "right_s":
                state_x += 1
            elif action == "up_s":
                state_x += 4
            else:
                state_x -= 4

        return self.S[state_x][state_y]

    def create_transition_function(self):
        # create an  empty dictionary of all the states possible
        self.T = {}
        for state_r in self.S:
            for state_c in state_r:
                for state in state_c:
                    # if state belongs to the env
                    if self.get_state_player(state) == 0:
                        action_dict = {}
                        for a in self.get_uncontrolled_actions():
                            action_dict.update({a: None})
                        self.T.update({(state.x, state.y, state.t): action_dict})
                    else:
                        action_dict = {}
                        for a in self.get_controlled_actions():
                            action_dict.update({a: None})
                        self.T.update({(state.x, state.y, state.t): action_dict})



    def get_transition_function(self):
        return self.T

    def set_transition_function(self, curr_state, action, next_state):
        self.T[(curr_state.x, curr_state.y, curr_state.t)][action] = next_state

    def create_atomic_propositions(self):

        # number of atomic proposition
        print("creating set of atomic proposition")

        # create set of pos_x and pos_y
        set_pos_x = set(f"x{a}" for a in self.env.pos_x)
        set_pos_y = set(f"y{b}" for b in self.env.pos_y)
        set_player_t = ("t0", "t1")
        self.AP = set((a, b, c) for a in set_pos_x for b in set_pos_y for c in set_player_t)

        # check to make sure the no. of atomic propositions id equal to N**2 x N**2 x 2
        assert len(self.AP) == int(self.env.pos_x.stop * self.env.pos_y.stop * len(set_player_t)), \
            f"Make sure number of atomic proposition is equals pos_x * pos_y * (t_0, t_ 1) " \
            f": {int(self.env.pos_x * self.env.pos_y * len(set_player_t))} "

    def get_atomic_propositions(self):
        return self.AP

    def print_transition_matrix(self):
        for k, v in self.T.items():
            # k is state and v a mapping of each action to the next state
            for action in v.keys():
                if v[action]:
                    print(f"State {k} transits to {action}, ({v[action].x}, {v[action].y}, {v[action].t})")

    # define a labelling function
    def labeling_function(self, state):
        """
        Labeling function is a mapping from each state to an atomic proposition
        :param state:
        :type state: state
        :return: an atomic proposition p
        :rtype: AP
        """
        # need to iterate through the set
        for ele in self.AP:
            if ele == (f"x{state.x}", f"y{state.y}", "t0" if state.t == 0 else "t1"):
                break
        return ele

    def _read_formula(self):
        """
        A function to prompt user to input a specification and use slugs to check if that winning condition
        is realizable for the constructed graph or not
        :return:
        :rtype:
        """
        if prompt_user:
            spec = input("Enter a winning formula")
        else:
            spec = "something gibberish for now"

        self.W.update({"spec": spec})

    def set_W(self, phi):
        self.W['spec'] = phi

    def get_W(self):
        return self.W

    def get_total_state_count(self):
        # S is 3D list
        count = 0
        for s_r in self.S:
            for s_c in s_r:
                for p in s_c:
                    count += 1

        return count

    def get_total_sys_state_count(self):
        # S_s is 1D list
        return len(self.S_s)

    def get_total_env_state_count(self):
        # S_s is 1D list
        return len(self.S_e)

    def get_total_init_state_count(self):
        # S_s is 1D list
        return len(self.Init)


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

    # assign states to their respective players
    game._initialize_states()

    # set fake player states
    # TODO replace this in future with some useful feature
    if set_fake_state_player:
        game.set_state_players()

    # method to print all the states in the environment
    if print_env_states:
        game.print_env_states()

    # set controlled and uncontrolled action tuples
    game.set_controlled_actions()
    game.set_uncontrolled_actions()

    # set init state
    game.set_initial_state()
    sample_state = game.S[5][1][0]
    game.set_state_actions(sample_state)

    # create set of atomic proposition
    game.create_atomic_propositions()
    # print(game.get_atomic_propositions())
    # print(len(game.AP))

    #testing labeling function
    state_label = game.labeling_function(state=sample_state)
    # add instance of labeling function to
    # game.W.update({"labeling_func": game.labeling_function})
    # print(game.W.get("labeling_func")(sample_state))
    # print(sample_state.action)
    # get controlled and uncontrolled action
    # print(game.get_initial_state().x, game.get_initial_state().y, game.get_initial_state().t)
    # print(game.get_state_actions(game.Init))

    game.create_transition_function()
    # if sample_state.t == 0:
    #     game.set_transition_function(sample_state, "up_e", game.S[0][0])
    # else:
    #     game.set_transition_function(sample_state, "up_s", game.S[0][0])
    # print(game.get_transition_function())

    return game

if __name__ == "__main__":
   main()