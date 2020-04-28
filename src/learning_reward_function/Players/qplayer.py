import numpy as np

class QPlayer:

    def __init__(self, transition_dict, player, decay, epsilon=0.05, gamma=0.9):
        self.decay = decay
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = 1
        # a flag to which agent this belongs to
        self.player = player
        # initialize the state value function
        self.numStates = len(transition_dict.keys())
        self.numSysActions = 5
        # self.V = np.ones(self.numStates)
        self.V = {}
        self.Q = {}
        # self.pi = np.ones((self.numStates, self.numSysActions))/ self.numSysActions
        self.pi = {}
        self.transition_dict = transition_dict
        self.learning = True

        # initialize pi, state Q table and state value function V
        self._initialize_pi()
        self._initialize_Q()
        self._initialize_V()

    def _initialize_V(self):
        """
        A function that maps a given state to its state-value function
        """
        # V is dictionary mapping from each state to its respective state-value
        for state in self.transition_dict.keys():
            self.V[state] = 1

    def _initialize_pi(self):
        """
        A method to initialize the policy function which is a mapping from a given state - action pair to its respective
        probability. pi is dictionary that maps each states and its respective action to its probability.
        IF there are no valid actions actions for a give state then pi[state] is a blank diction
        """
        # the trasition dict is a dict {State 0 : {"Up" : next_state , "Down": next_state, "Right": None,},{...},{.}}}
        for state, actions in self.transition_dict.items():
            self.pi[state] = {}
            # numOfValidActions = 0
            valid_action_list = []
            for action, child_state in actions.items():
                if child_state is not None:
                    # numOfValidActions += 1
                    valid_action_list.append(action)

            # if numOfValidActions != 0:
            for a in valid_action_list:
                self.pi[state].update({a: 0})

    def _initialize_Q(self):
        """
        A function to initialize Q value which is a mapping from a given state and a pair of system and env robot action
        to its quality.
        """
        # create a dictionary of the formal Q[state][sys_action][env_action]
        for state in self.transition_dict.keys():
            # which player does the current state belong to
            # t = state[2]
            self.Q[state] = {}
            # actions = []

            #  0 - sys player
            if self.player == 0:
                actions = self._get_valid_actions(0, state)
            else:
                actions = self._get_valid_actions(1, state)
            # for p in [0, 1]:
            #     if p == 0:
            #         # sys_action = self._get_valid_actions(p, state)
            #         actions.append(self._get_valid_actions(p, state))
            #     else:
            #         # env_action = self._get_valid_actions(p, state)
            #         actions.append(self._get_valid_actions(p, state))
            for valid_action in actions:
                self.Q[state].update({valid_action: 1})

                # # the first set belongs to the system agent
                #
                # self.Q[state][valid_action] = {}
                # # the second set belongs to the env agent
                # for valid_env_action in actions[1]:
                #     self.Q[state][valid_sys_action].update({valid_env_action: 1})

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

    def _get_best_action(self, state):
        q_value_of_state = self.Q[state]
        max_value = max(q_value_of_state.values())
        # choose an action randomly if there multiple actions with
        action = np.random.choice([k for k, v in q_value_of_state.items() if v == max_value])
        return action


    def chooseAction(self,player, state):
        # get the available actions for a give states from the transition system
        available_actions = self._get_valid_actions(player, state)
        if self.learning and np.random.rand() < self.epsilon:
            # randomly choose an action
            action = np.random.choice(available_actions)
        else:
            action = self._get_best_action(state)
        return action

    def learn(self, initialState, finalState, action, reward):
        if not self.learning:
            return

        self.Q[initialState][action] = (1 - self.alpha) * self.Q[initialState][action] + \
            self.alpha * (reward + self.gamma * self.V[finalState])

        # bestAction = np.argmax(self.Q[initialState])
        bestAction = self._get_best_action(initialState)
        # self.pi[initialState] = np.zeros(self.numActions)
        self.pi[initialState][bestAction] = 1
        self.V[initialState] = self.Q[initialState][bestAction]
        self.alpha *= self.decay


if __name__ == "__main__":
    print("***********************main file to implement q learning player agent***********************")