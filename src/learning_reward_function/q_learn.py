import pylab as pl
import numpy as np
import operator
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as anim
import matplotlib.image as mping
import re
import warnings
import sys
import math

from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from src import main

# flag to plot while training
plot_while_training = False
# flaf to plot only after training
plot_after_training = False
# flag to add robot as patch
use_robot_as_patch = True

class GridWorld:

    def __init__(self, height, width, sys_init, A_c=None, A_uc=None):
        self.height = height
        self.width = width
        self.controlled_actions = A_c
        self.uncontrolled_actions = A_uc
        self.init_state = sys_init
        # -1 because thats the default reward in each state
        self.grid = np.zeros((self.height, self.width)) - 1
        # randomly select an initial state
        # sys_init is a list of valid initial states
        self.initPosition = self._get_random_init_state()

        self.current_location = self.initPosition

        # set location for bomb and the gold
        self.bomb_location = (1, 3)
        self.gold_location = (0, 3)
        self.terminal_states = [self.bomb_location, self.gold_location]

        # set grid rewards for special cells
        self.grid[self.bomb_location[0], self.bomb_location[1]] = -10
        self.grid[self.gold_location[0], self.gold_location[1]] = 10

        # actions avaialble to the agent
        # self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def _get_random_init_state(self):
        rand_init_state = np.random.choice(self.init_state)
        return (rand_init_state.x, rand_init_state.y)

    def restart(self, height=None, width=None, sys_init=None, env_init=None):
        if sys_init is not None:
            self.initPosition = sys_init
        else:
            self.initPosition = self._get_random_init_state()

        # if env_init is not None:
        #     self.initPosition[1] = env_init

    def get_available_sys_action(self):
        return self.controlled_actions

    def get_available_env_actions(self):
        return self.uncontrolled_actions

    def agent_on_map(self):
        grid_map = np.zeros((self.height, self.width))
        grid_map[self.current_location[0], self.current_location[1]] = 1
        return grid_map
    # write helper method to convert pos_to_xy map
    def pos_to_xy_map(self, location):
        """
        A method to convert location of format (pos x pos x player) to (x1,y1) , (x2,y2)
        :param location: a tuple of type (pos x pos x player)
        :type location: tuple
        :return: A tuple containing the system position and env position
        :rtype: tuple
        """
        sys_pos = location[0]  # an integer value of a cell
        env_pos = location[1]  # an integer value of a cell

        # assuming that we are in a gridworld n = m and max(cell_value) = n**2 -1
        # get original env width(n) and height(m)
        org_n = int(math.sqrt(self.height))
        # since its a square world
        org_m = org_n
        # convert pos to x,y
        # finding a value of x and y that satifies pos = n*y + x
        for x in range(org_n):
            for y in range(org_m):
                if org_n*y + x == sys_pos:
                    sys_xy = (x, y)
                if org_n*y + x == env_pos:
                    env_xy = (x, y)

        return (sys_xy, env_xy)

    def get_reward(self, new_location):
        # reward is +1 is the both the robot are in diagonally opposite position to each other else its zero
        # get the sys and env xy values
        sys_xy, env_xy = self.pos_to_xy_map(new_location)
        # system xy coordinates
        x1 = sys_xy[0]
        y1 = sys_xy[1]
        # env xy coordinates
        x2 = env_xy[0]
        y2 = env_xy[1]
        # two points X, Y are diagonally opposite if abs(x1 - x2) = abs(y1 - y2)
        if abs(x1 - x2) == abs(y1 - y2):
            return 1
        else:
            return 0

    def make_step(self, action):
        """Moves the agent in the specified direction. If agent is at a border, agent stays still
        but takes negative reward. Function returns the reward for the move."""
        # Store previous location
        last_location = self.current_location

        # A_c : [up_s, down_s, left_s, right_s, stay_s]
        # A_uc: [up_e, down_e, left_e, right_e]
        up_action = re.compile('^up')
        down_action = re.compile('^down')
        left_action = re.compile('^left')
        right_action = re.compile('^right')
        stay_action = re.compile('^stay')

        # UP
        if up_action.match(action):
            # agent is at the bottom of the cell
            # self.current_location = (self.current_location[0], self.current_location[1] + 1)
            self.current_location = (self.current_location[0] + 4, self.current_location[1])
        # DOWN
        elif down_action.match(action):
            # self.current_location = (self.current_location[0], self.current_location[1] - 1)
            self.current_location = (self.current_location[0] - 4, self.current_location[1])
        # LEFT
        elif left_action.match(action):
            self.current_location = (self.current_location[0] - 1, self.current_location[1])
            # self.current_location = (self.current_location[0] - 1, self.current_location[1])
        # RIGHT
        elif right_action.match(action):
            self.current_location = (self.current_location[0] + 1, self.current_location[1])
            # self.current_location = (self.current_location[0] + 1, self.current_location[1])
        # STAY
        elif stay_action.match(action):
            self.current_location = (self.current_location[0], self.current_location[1])
            # self.current_location = self.current_location
        # Not a valid action
        else:
            warnings.warn("Encountered a invalid action. This should have never occured. Exiting progrem")
            sys.exit(1)

        reward = self.get_reward(self.current_location)

        return reward

    def check_state(self):
        if self.current_location in self.terminal_states:
            return 'TERMINAL'

class Q_Agent:

    def __init__(self, env, transition_dict ,epsilon=0.05, alpha=0.1, gamma=1):
        self.environment = env
        self.q_table = dict()
        self.transition_dict = transition_dict
        # for x in range(env.height):
        #     for y in range(env.width):
        #         self.q_table[(x,y)] = {'UP': 0, 'DOWN': 0, 'LEFT': 0, 'RIGHT': 0}

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self._update_q_table()

    def _update_q_table(self):
        for x in range(env.width):
            for y in range(env.height):
                # only populating states that belong to system player

                self.q_table[(x, y)] = {a: 0 for a, a_v in self.transition_dict[(x, y, 1)].items() if a_v is not None}


    def choose_action(self, available_actions):
        if np.random.uniform(0, 1) < self.epsilon:
            action = available_actions[np.random.randint(0, len(available_actions))]
        else:
            q_vales_of_state = self.q_table[self.environment.current_location]
            max_value = max(q_vales_of_state.values())
            action = np.random.choice([k for k, v in q_vales_of_state.items() if v == max_value])

        return action

    def learn(self, old_state, reward, new_state, action):
        """Update the Q value table using Q-learning"""
        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[old_state][action]

        self.q_table[old_state][action] = ((1- self.alpha)*current_q_value + \
                                          self.alpha * (reward + (self.gamma * max_q_value_in_new_state)))


def play(env, agent, trials=500, max_steps_per_episode=1000, learn=False):
    reward_per_episode = []
    # initialize fig handle as None
    fig = None
    # load robot_patch if need be
    if use_robot_as_patch:
        robot_patch = mping.imread("misc_files/robot_emoji_1.jpg")

    for trial in range(trials):
        if plot_while_training:
            # plot the basic gridworld
            if fig is None:
                fig, ax = plot_gridworld_with_labels(env, num=f"Iteration # {trial}")
            else:
                plt.close()
                fig, ax = plot_gridworld_with_labels(env, num=f"Iteration # {trial}")
        cumulative_reward = 0
        step = 0
        game_over = False
        while step < max_steps_per_episode and game_over is not True:
            old_state = env.current_location
            # pass in only available actions at a give state
            available_action = [k for k, v in agent.transition_dict[(old_state[0], old_state[1], 1)].items() if v is not None]
            # for k, v, in agent.transition_dict[()]
            action = agent.choose_action(available_action)
            reward = env.make_step(action)
            new_state = env.current_location

            if learn is True:
                agent.learn(old_state, reward, new_state, action)

            cumulative_reward += reward
            if plot_while_training:
                # plot after each step
                plot_agent_on_gridworld(ax, env.current_location, image=robot_patch)

            step += 1

            if env.check_state() == 'TERMINAL':
                # hardcoding these values for now but need to add a reset method in the env class
                env.restart()
                game_over = True

        reward_per_episode.append(cumulative_reward)

    return reward_per_episode

# roll out the policy it lerned
def roll_out_policy(agent, env):

    curr_state = env.current_location
    final_reward = 0
    # plot the basic gridworld
    fig, ax = plot_gridworld_with_labels(env)
    iter = 1
    if use_robot_as_patch:
        robot_patch = mping.imread("misc_files/robot_emoji_1.jpg")

    # find the best action at the currest state, execute it do this repeatedly
    while 1 and iter < 10:
        iter += 1
        # select action a with max q value at S
        q_value_of_the_state = agent.q_table[curr_state]
        max_value = max(q_value_of_the_state.values())
        action = np.random.choice([k for k, v in q_value_of_the_state.items() if v == max_value])

        # use this action to transit to next state
        reward = env.make_step(action)
        curr_state = env.current_location
        # plot the final roll out policy
        if plot_after_training:
            plot_agent_on_gridworld(ax, env.current_location, pause_time=0.1, image=robot_patch)

        final_reward += reward

        if env.check_state() == 'TERMINAL' or iter == 10:
            print("Reached the final state")
            plt.close()
            fig, ax = plot_gridworld_with_labels(env)
            plotQ(ax, env, agent.q_table, plot_square)
            break

def load_robot_patch(xy ,robot_patch):
    # lets try loading robot emoji as our controlled robot system
    # robot_patch = mping.imread(file_name)
    imageBox = OffsetImage(robot_patch, zoom=0.1)
    ab = AnnotationBbox(imageBox, xy, frameon=False)
    return ab

# def plot_labeled_arows(dirs):
#     def draw(ax, xy, qs):
#
#         color_square(ax, xy, cm.coolwarm(max(qs)))
#
#         for q, direction in zip(qs, dirs):
#             v = (q - min(qs))/ max(qs) - min(qs)
#
#             if not all(direction == CENTER):
#                 plot_arrow()
#             else:
#                 plot_square()
#
#     return draw

def normalize(qs):
    # get the max and min q value
    allq = [q for qs in qs.values() for q in qs.values()]
    vmax, vmin = max(allq), min(allq)

    range = max(vmax - vmin, 1e-8)

    def norm(q):
        return (q - vmin)/range

    return {s: map(norm, a_value) for s, qs_values in qs.items() for a, a_value in qs_values.items()}


def plot_square(ax, xy, offset, color='green'):
    x, y, = xy
    # offset = offset * .47
    # x = x + offset[0]*.15
    # y = y + offset[1]*.15

    ax.add_patch(patches.Rectangle(
        (x, y),
        0.1,
        0.1,
        facecolor=color,
        edgecolor='black',
        alpha=0.5
    ))

def plot_arrow(ax, xy, action, color='blue'):
    x,  y = xy
    if action == "RIGHT":
        # draw upwards as left is upwards
        dx = 0
        dy = 0.3
    elif action == "LEFT":
        # draw downwards
        dx = 0
        dy = -0.3
    elif action == "UP":
        # draw left <---
        dx = -0.3
        dy = 0
    else:
        # draw right
        dx = 0.3
        dy = 0

    ax.add_patch(patches.Arrow(
        x, y, dx=dx, dy=dy, width=.3, facecolor=color
    ))


def get_best_Action(curr_state, q_table):
    action = None
    q_value_of_the_state = q_table[curr_state]
    if len(q_value_of_the_state.keys()) != 0:
        max_value = max(q_value_of_the_state.values())
        action = np.random.choice([k for k, v in q_value_of_the_state.items() if v == max_value])
    return action


def plotQ(ax, gridworld, qs, plot_square):
    # get the normalized values
    # qs = normalize(qs)

    for x in range(gridworld.width):
        for y in range(gridworld.height):
            xy = gridworld_to_xy(x, y)
            # plot_square(ax, xy, qs[(x, y)])
            color_square(ax, xy, color="green")
            # find the optimal action and draw an arrow in that direction
            best_action = get_best_Action(curr_state=(x, y), q_table=qs)
            if best_action:
                plot_arrow(ax, xy, best_action)
                plt.draw()
                plt.pause(0.1)

    plt.show()

def plot_agent_on_gridworld(ax, agent_xy, color="green", pause_time=0.01, **kwargs):
    (x, y) = gridworld_to_xy(agent_xy[0], agent_xy[1])
    robo_image = kwargs['image']
    if use_robot_as_patch:
        # # lets try loading robot emoji as our controlled robot system
        # robot_patch = mping.imread('robot_emoji_1.jpg')
        # imageBox = OffsetImage(robot_patch, zoom=0.1)
        # ab = AnnotationBbox(imageBox, (x, y), frameon=False)

        ax.add_artist(load_robot_patch((x,y),robo_image))
    else:
        # plot a circular agent in the girdworld
        ax.add_patch(patches.Circle(
            (x, y),
            radius=0.2,
            color=color,
            alpha=1  # transparency value
        ))

        # ax.add_patch(patches.CirclePolygon(
        #     (x, y),
        #     radius=0.25,
        #     color=color,
        #     alpha=1,
        # ))

        # ax.add_patch(plt.Circle(
        #     (x, y),
        #     radius=0.25,
        #     color=color,
        #     alpha=1,
        # ))
    ax.set_aspect('equal')
    plt.draw()
    plt.pause(pause_time)


def grid_plot(env, *args):
    fig_title = args[0]
    fig = pl.figure(num=fig_title)
    ax = fig.add_subplot(111)

    cmax = env.width
    rmax = env.height

    ax.set_xticks(range(cmax + 1))
    ax.set_yticks(range(rmax + 1))
    pl.grid(True)
    return fig, ax

def plot_num(ax, xy, offset, value, **kwargs):
    x, y = xy
    # offset = offset * .47
    # textxy = (x + offset[0] * 0.85, y + offset[1] * 0.85)

    ax.annotate(value, xy=(x, y), xycoords='data',
                horizontalalignment='center',
                verticalalignment='center', **kwargs)

def gridworld_to_xy(grid_x, grid_y):
    return (grid_x + 0.5, grid_y + 0.5)

# add the bomb and  the gold biscuit box path
def color_square(ax, xy, color):
    x, y = xy

    ax.add_patch(patches.Rectangle(
        (x - 0.5, y - 0.5),
        0.99,  # width
        0.99,  # height
        color=color,
        alpha=0.5  # transparency value
    ))

def plot_gridworld_with_labels(env, **kwargs):

    plot_title = kwargs.get('num', 1)
    fig, ax = grid_plot(env, plot_title)
    # lets annotate the gridworld
    for x in range(env.width):
        for y in range(env.height):
            xy_coord = gridworld_to_xy(x, y)
            plot_num(ax, xy_coord, np.array((1, 1)), f'{x, y}')

            if (x, y) == env.bomb_location:
                color_square(ax, xy_coord, (85 / 255, 120 / 255, 0 / 255))

            if (x, y) == env.gold_location:
                color_square(ax, xy_coord, (255 / 255, 191 / 255, 0 / 255))

            # plt.draw()
            # plt.pause(0.01)
    return fig, ax


if __name__ == "__main__":

    game = main.construct_game()
    x_length = game.env.env_data["env_size"]["n"]
    y_length = game.env.env_data["env_size"]["m"]
    slugsfile_path = "../two_player_game/slugs_file/CoRL_1"
    sys_str, _ = main.extract_permissive_str(slugsfile_path)
    main.updatestratetegy_w_state_pos_map(sys_str, x_length, y_length)
    main.pre_processing_of_str(sys_str, x_length, y_length)
    width = game.env.pos_x.stop
    height = game.env.pos_y.stop

    G_hat = main.create_G_hat(strategy=sys_str, game=game)
    env = GridWorld(height, width, G_hat.Init)
    agent = Q_Agent(env, G_hat.T)

    reward_per_episode = play(env, agent, trials=500, learn=True)
    print("Done training \n")
    roll_out_policy(agent, env)
    # plot the final policy learned
    plt.plot(reward_per_episode)
    plt.show()

