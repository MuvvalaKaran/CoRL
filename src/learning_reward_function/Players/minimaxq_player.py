import numpy as np
import pulp
import sys

class MinimaxQPlayer:

    def __init__(self, transition_dict,  decay, epsilon=0.05, gamma=0.9):
        self.decay = decay
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = 1
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

    def _initialize_pi(self):
        """
        A method to initialize the policy function which is a mapping from a given state - action pair to its respective
        probability. pi is dictionary that maps each states and its respective action to its probability.
        IF there are no valid actions actions for a give state then pi[state] is a blank diction
        """
        # the trasition dict is a dict {State 0 : {"Up" : next_state , "Down": next_state, "Right": None,},{...},{.}}}
        for state, actions in self.transition_dict.items():
            self.pi[state] = {}
            numOfValidActions = 0
            valid_action_list = []
            for action, child_state in actions.items():
                if child_state is not None:
                    numOfValidActions += 1
                    valid_action_list.append(action)

            if numOfValidActions != 0:
                for a in valid_action_list:
                    self.pi[state].update({a: 1/numOfValidActions})

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
            actions = []
            for p in [0, 1]:
                if p == 0:
                    # sys_action = self._get_valid_actions(p, state)
                    actions.append(self._get_valid_actions(p, state))
                else:
                    # env_action = self._get_valid_actions(p, state)
                    actions.append(self._get_valid_actions(p, state))
            for valid_sys_action in actions[0]:
                # the first set belongs to the system agent
                self.Q[state][valid_sys_action] = {}
                # the second set belongs to the env agent
                for valid_env_action in actions[1]:
                    self.Q[state][valid_sys_action].update({valid_env_action: 1})

    def _initialize_V(self):
        """
        A function that maps a given state to its state-value function
        """
        # V is dictionary mapping from each state to its respective state-value
        for state in self.transition_dict.keys():
            self.V[state] = 1

    def _get_Q_value(self, state):
        raise NotImplementedError

    def _get_available_Actions(self, player, state):
        raise NotImplementedError
        # create an empty list to hold all the available actions for a give player
        # available_actions = []
        # # 0 is sys_player and 1 is env player
        # if player == 0 :
        #     t = 1
        # else:
        #     t = 0
        # for k, v in self.transition_dict(state[0], state[1], t):
        #     if v is not None:
        #         available_actions.append(k)
        # return available_actions

    def _get_valid_actions(self ,player, state):
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
        q_value_of_state = self._get_Q_value(state)
        max_value = max(q_value_of_state)
        # choose an action randomly if there multiple actions with
        action = np.random.choice([k for k, v in q_value_of_state.items() if v == max_value])

        return action

    # we will implement a epsilon-greedy algorithm to choose our action
    def chooseAction(self, player, state):
        # get the available actions for a give states from the transition system
        available_actions = self._get_valid_actions(player, state)
        if self.learning and np.random.rand() < self.epsilon:
            # randomly choose an action
            action = np.random.choice(available_actions)
        else:
            action = self._weighted_action_choice(state)
        return action

    def _weighted_action_choice(self, state):
        rand = np.random.rand()
        cum_sum_prob = np.cumsum(list(self.pi[state].values()))
        action = 0
        while rand > cum_sum_prob[action]:
            action += 1
        # return the corresponding action string e,g "Up_s"
        lst = list(self.pi[state].keys())
        return lst[action]

    def learn(self, currentState, newState, actions, reward):
        if not self.learning:
            return

        sys_action, env_action = actions
        self.Q[currentState][sys_action][env_action] = (1 - self.alpha) * self.Q[currentState][sys_action][env_action] + \
                                                       self.alpha * (reward + self.gamma * self.V[newState])
        self.V[currentState] = self.updatePolicy(currentState)
        self.alpha *= self.decay

    def updatePolicy(self, state):
        # compute pi and then update Value of the state
        self.pi[state] = self._compute_pi(state=state)
        # get env actions
        EnvActions = self._get_valid_actions(player=1, state=state)
        SysActions = self._get_valid_actions(player=0, state=state)
        min_expected_value = sys.maxsize

        for env_action in EnvActions:
            expected_value = sum([self.pi[state][sys_action] * self.Q[state][sys_action][env_action] for sys_action in SysActions])

            if expected_value < min_expected_value:
                min_expected_value = expected_value

        return min_expected_value

    def _compute_pi(self, state):

        # get available actions for system player
        # numSysActions = len(self._get_valid_actions(player=0, state=state))
        SysActions = self._get_valid_actions(player=0, state=state)
        EnvActions = self._get_valid_actions(player=1, state=state)
        numEnvActions = len(EnvActions)
        numSysActions = len(SysActions)
        pi = pulp.LpVariable.dict("pi", range(numSysActions), 0, 1)
        max_min_value = pulp.LpVariable("max_min_value")
        lp_prob = pulp.LpProblem("Maxmin Problem", pulp.LpMaximize)
        lp_prob += (max_min_value, "objective")

        lp_prob += (sum([pi[i] for i in range(numSysActions)]) == 1)

        for env_action in EnvActions:
            label = "{}".format(env_action)
            values = pulp.lpSum([pi[idx] * self.Q[state][sys_action][env_action] for idx, sys_action in enumerate(SysActions)])
            condition = max_min_value <= values
            lp_prob += condition

        lp_prob.solve()

        # return the respective action and action probability value
        action_dict = {}
        for i in range(len(pi.keys())):
            action_dict.update({SysActions[i] : pi[i].value()})

        return action_dict

    def policyForState(self):
        raise NotImplementedError

if __name__ == "__main__":
    print("***********************main file to implement minimaxq player agent***********************")