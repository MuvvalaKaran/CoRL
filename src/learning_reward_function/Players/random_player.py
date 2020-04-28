import numpy as np

class RandomPLayer:

    def __init__(self, transition_dict):
        self.transition_dict = transition_dict

    def _get_valid_actions(self, player, state):
        """
                A helper method that returns a list of valid actions for a given state and a player. e.g Say (x, y, 1) is the
                current state and we want the set of valid  actions for sys player. We return the set of valid action from the
                transition_dict for (x, y, 1) for the env player we return the set of valid actions from the (x, y, 0).

                Note S = (Pos x Pos x t) where t = 1 system player and t = 0 env player. But the player variable above is
                flipped.
                :param player:
                :type player: int
                :param state:
                :type state: tuple
                :return:
                :rtype:
                """
        # create an empty list to hold all the available actions for a give player
        available_actions = []
        # 0 is sys_player and 1 is env player
        if player == 0:
            t = 1
        else:
            t = 0
        for k, v in self.transition_dict[(state[0], state[1], t)].items():
            if v is not None:
                available_actions.append(k)
        return available_actions

    def chooseAction(self, player, state):
        available_actions = self._get_valid_actions(player, state)
        return np.random.choice(available_actions)

    def learn(self, currentState, newState, actions, reward):
        pass