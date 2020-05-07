# parent script file
# 1. Construct a game G and Extract a permissive str using slugs
# 2. Check if the specification is realizable and if so apply the str to G to compute G_hat
# 3. Compute the most optimal str that maximizes the reward while also satisfying the safety and liveness guarantees
# 4. Map the optimal str back to G
import copy
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import re
import sys
import matplotlib.patches as patches
import matplotlib.image as mping
import pickle
import os
import imageio
import time
import yaml

from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from learning_reward_function.Players.minimaxq_player import MinimaxQPlayer as minimax_agent
from learning_reward_function.Players.random_player import RandomPLayer as rand_agent
from learning_reward_function.Players.qplayer import QPlayer as q_agent
from learning_reward_function.MakovGame.diagonal_players import Gridworld
from two_player_game import TwoPlayerGame
from two_player_game import extractExplicitPermissiveStrategy

# flag to plot BDD diagram we get from the slugs file
plot_graphs = False
# flag to print length of child nodes
print_child_node_length = False
# flag to print strategy after preprocessing
print_str_after_processing = False
# flag to print state num and state_pos_map value
print_state_pos_map = False
# flag to print the update transition matrix
print_updated_transition_function = False

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""
    def newFunc(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc


class plotterClass():

    def __init__(self, fig_title, robo_file_path=None, xlabel=None, ylabel=None):
        self.fig = None
        self.ax = None
        self.fig_title = fig_title
        self.xlabel = xlabel
        self.ylabel = ylabel

        # initialize a figure an axes handle
        self._create_ax_fig()
        if robo_file_path is not None:
            self.robot_patch = mping.imread(robo_file_path)

    def _create_ax_fig(self):
        self.fig = pl.figure(num=self.fig_title)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)


    def plot_policy(self, env, sys_player, sys_str):
        width = env.env.pos_x.stop
        height = env.env.pos_y.stop

        self.plot_game_world(width, height, print_grid_num=True, fontsize=self._font_size_dict(str(width)))
        # current position is a tuple of the format (x, y, t=1)

        # for x in env.env.pos_x:
        #     for y in env.env.pos_y:
        for k, v in sys_str.items():
            x = v['state_pos_map'][0]
            y = v['state_pos_map'][1]
            t = int(v['state_pos_map'][2])
            if t == 1:
                sys_best_action = sys_player.chooseAction(player=0, state=(x, y, t))
                xy = (x, y)
                # color square as light green
                self._add_patch(shape='rect', xy=xy, color='green')
                self._add_patch(shape='arrow', xy=xy, action=sys_best_action, color='blue')

    def plot_state_value(self, env, sys_player):
        # plot the game world
        width = env.env.pos_x.stop
        height = env.env.pos_y.stop
        self.plot_game_world(width, height, print_grid_num=False, fontsize=self._font_size_dict(str(width)))

        offset = 0.5
        # for each state in V print the number on the respective grid
        for k, v in sys_player.V.items():
            # only concerned with the system state values
            if k[2] == 1:
                x = k[0]
                y = k[1]
                # color square as light green
                self._add_patch(shape='rect', xy=(x, y), color='green')
                # annotate the box with the state value
                self.ax.annotate(f"{v:.3f}", xy=(offset + x, offset + y), xycoords='data',
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=self._font_size_dict(str(width)))

    def plot_q_value(self, env, sys_player):
        # plot the game world
        width = env.env.pos_x.stop
        height = env.env.pos_y.stop
        self.plot_game_world(width, height, print_grid_num=False, fontsize=self._font_size_dict(str(width)))
        offset = 0.5
        for k, v in sys_player.pi.items():
            # only concerned with the system state values
            if k[2] == 1:
                x = k[0]
                y = k[1]
                # color square as light green
                self._add_patch(shape='rect', xy=(x, y), color='green')
                for action, action_value in v.items():
                    # only plot those action that have probability of choosing as 1.
                    if action_value == 1:
                        # draw arrows for each transition
                        if action == "stay_s":
                            self.ax.annotate(f"stay", xy=(offset + x, offset + y), xycoords='data',
                                             horizontalalignment='center',
                                             verticalalignment='center', fontsize=self._font_size_dict(str(width)))
                        else:
                            self._add_patch(shape="arrow", action=action, xy=(x, y), color='blue')

    def _font_size_dict(self,  grid_size):
        """
        Helper method to map a given grid size to its respective font size
        :param grid_size: N**2 value
        :type grid_size: basestring
        :return: fontsize
        :rtype: int
        """
        fontsize_dict = {
            '9': 10,
            '16': 6,
            '25': 3,
            '36': 2,
            '49': 1,
        }

        return fontsize_dict[grid_size]

    def plot_game_world(self, width, height, print_grid_num=True, fontsize=None):
        # flag to print the grid numbers
        if print_grid_num:
            # we need to offset both the x and y to print in the center of the grid
            for x in range(width):
                for y in range(height):
                    off_x, off_y = x + 0.5, y + 0.5
                    self.plot_grid_num((off_x, off_y), value=f"{x, y}", fontsize=fontsize)

                    # add obstacles to the gridworld when n = 7
                    if width == 49:
                        # list containing the cell number of obstacle to be plotted
                        obs_cell_list = [2, 3, 4, 9, 31, 36, 37, 38]
                        # add black patch to these cells
                        if y in obs_cell_list:
                            self._add_obstables((x, y))

                        if x in obs_cell_list:
                            self._add_obstables((x, y))

        # the location of the x_ticks is the middle of the grid
        def offset_ticks(x, offset=0.5):
            return x + offset

        # ticks = locations of the ticks and labels = label of the ticks
        self.ax.set_xticks(ticks=list(map(offset_ticks, range(width))), minor=False)
        self.ax.set_xticklabels(labels=range(width))
        self.ax.xaxis.set_tick_params(labelsize=self._font_size_dict(str(width)))
        self.ax.set_yticks(ticks=list(map(offset_ticks, range(width))), minor=False)
        self.ax.set_yticklabels(labels=range(width))
        self.ax.yaxis.set_tick_params(labelsize=self._font_size_dict(str(width)))

        # add points for gridline plotting
        self.ax.set_xticks(ticks=range(width), minor=True)
        self.ax.set_yticks(ticks=range(height), minor=True)

        # set x and y limits to adjust the view
        self.ax.set_xlim(left=0, right=width)
        self.ax.set_ylim(bottom=0, top=height)

        # set the gridlines at the minor xticks positions
        self.ax.yaxis.grid(True, which='minor')
        self.ax.xaxis.grid(True, which='minor')

    def plot_gridworld(self, env, print_grid_num=False, fontsize=None,  *args):
        # the env here the original env of size (N x M)
        # fig_title = args[0]

        cmax = env.env.env_data["env_size"]["n"]
        rmax = env.env.env_data["env_size"]["m"]

        # flag to print the grid numbers
        if print_grid_num:
            # we need to offset both the x and y to print in the center of the grid
            for x in range(cmax):
                for y in range(rmax):
                    off_x, off_y = x + 0.5, y + 0.5
                    self.plot_grid_num((off_x, off_y), value=f"{x, y}", fontsize=fontsize)

                    # add obstacles to the gridworld when n = 7
                    if cmax == 7:
                        # add the obstacle - 1 patch to the gridworld
                        if y == 0 and (5 > x > 1):
                            self._add_obstables((x, y))

                        if y == 1 and x == 2:
                            self._add_obstables((x, y))

                        if y == 4 and x == 3:
                            self._add_obstables((x, y))

                        # add the obstacle -2 patch to the gridworld
                        if y == 5 and (0 < x < 4):
                            self._add_obstables((x, y))

        # the location of the x_ticks is the middle of the grid
        def offset_ticks(x, offset=0.5):
            return x + offset
        # ticks = locations of the ticks and labels = label of the ticks
        self.ax.set_xticks(ticks=list(map(offset_ticks, range(cmax))), minor=False)
        self.ax.set_xticklabels(labels=range(cmax))
        self.ax.set_yticks(ticks=list(map(offset_ticks, range(rmax))), minor=False)
        self.ax.set_yticklabels(labels=range(rmax))

        # add points for gridline plotting
        self.ax.set_xticks(ticks=range(cmax), minor=True)
        self.ax.set_yticks(ticks=range(rmax), minor=True)

        # set x and y limits to adjust the view
        self.ax.set_xlim(left=0, right=cmax)
        self.ax.set_ylim(bottom=0, top=rmax)

        # set the gridlines at the minor xticks positions
        self.ax.yaxis.grid(True, which='minor')
        self.ax.xaxis.grid(True, which='minor')

    @deprecated
    def _fig2data(self):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        self.fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = self.fig.canvas.get_width_height()
        buf = np.fromstring(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        return buf


    def plot_grid_num(self,xy, value, offset=None, **kwargs):
        """
        A method to add text to the grid world
        :param xy: (x, y) position of the text. No offset included in here
        :type xy: tuple
        :param value: The text that should go at each block
        :type value: basestring
        :param offset: how much to offset the values by in both x and y direction
        :type offset: int
        :param kwargs:
        :type kwargs:
        """
        x, y, = xy
        self.ax.annotate(value, xy=(x, y), xycoords='data',
                    horizontalalignment='center',
                    verticalalignment='center',
                    **kwargs)

    def save_fig(self, name, dpi=200, format='png'):
        """
        A method to save the figure plotted in a given directory within src/
        :param name: should be of the format two_player_game/plots/name_of_the_plot.png
        :type name: basestring
        :param dpi: Dots per inches (dpi) determines how many pixels the figure comprises. The default dpi in
        matplotlib is 100.
        :type dpi: int
        """
        plt.savefig(name, dpi=dpi, format=format)

    def _add_patch(self, shape, xy, color='green', robo_image=None, action=None, alpha=0.5):
        x, y = xy
        shape_case = {
            'circle': 1,
            'rect': 2,
            'arrow': 3,
            'robot': 4
        }

        if shape_case[shape] == 1:
            circle = patches.Circle(
                (x, y),
                radius=0.2,
                color=color,
                alpha=1  # transparency value
            )
            self.ax.add_patch(circle)

            return circle
        elif shape_case[shape] == 2:
            rect = patches.Rectangle(
                (x, y),
                0.99,
                0.99,
                facecolor=color,
                edgecolor='black',
                alpha=alpha  # transparency value
            )
            self.ax.add_patch(rect)

            return rect
        elif shape_case[shape] == 3:
            # A_c : [up_s, down_s, left_s, right_s, stay_s]
            def plot_arrow():
                # create an arrow starting form x,y and extending in dx and dy accordingly
                off = 0.5
                arrow = patches.Arrow(
                    off + x, off + y, dx=dx, dy=dy, width=.3, facecolor=color
                )

                self.ax.add_patch(arrow)

            if action == "right_s":
                dx = 0.3
                dy = 0

                plot_arrow()
            elif action == "up_s":
                dx = 0
                dy = 0.3

                plot_arrow()
            elif action == "down_s":
                dx = 0
                dy = -0.3

                plot_arrow()
            elif action == "left_s":
                dx = -0.3
                dy = 0

                plot_arrow()
            else:
                pass
                # self.ax.text(x + 0.5, y + 0.5, "stay action", fontsize=7, transform=self.ax.transAxes)

        elif shape_case[shape] == 4:
            imageBox = OffsetImage(robo_image, zoom=0.1)
            ab = AnnotationBbox(imageBox, xy, frameon=False)
            self.ax.add_artist(ab)
        else:
            return warnings.warn("Not a valid shape. Need to add that patch to the _add_patch function")

    def _add_obstables(self, xy):
        # add a black patch to the square that belongs to obstacle at location x, y
        self._add_patch('rect', xy, color='black')

    def _convert_pos_to_xy(self, location, n, m):
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
        # self.org_n = int(math.sqrt(self.height))
        # # since its a square world
        # self.org_m = self.org_n

        # convert pos to x,y
        # finding a value of x and y that satifies pos = n*y + x
        for x in range(n):
            for y in range(m):
                if n * y + x == sys_pos:
                    sys_xy = (x, y)
                if n * y + x == env_pos:
                    env_xy = (x, y)

        if isinstance(sys_xy, type(None)) or isinstance(env_xy, type(None)):
            warnings.warn("Could not convert the pos value of system or env robot to its respective (x,y) values. " \
                          "Ideally this should have never occurred. The system pos value is" + sys_pos + "and the env " \
                          "pos values is" + env_pos + ". Exiting code")
            sys.exit(-1)

        return sys_xy, env_xy

    def roll_out_final_policy(self, env, init_state_list, sys_player, env_player,
                              game,
                              plot_title="Final Policy Evaluation",
                              print_grid_num=False,
                              sys_init=None,
                              env_init=None,
                              ep_len=30):
        """
        A method to visualize the learned policy.
        :param env: Is the original env in the twoplayergame folder
        :type env: @ gridWorld
        :param init_state:
        :type init_state: list
        :param sys_player:
        :type sys_player: @q_agent
        :param env_player:
        :type env_player: @q_agent
        :param game:
        :type game: @gridworld
        :param plot_title:
        :type plot_title: basestring
        :param print_grid_num:
        :type print_grid_num: bool
        :param sys_init:
        :type sys_init: int
        :param env_init:
        :type env_init: int
        """
        # init_state is set og valid initial states of the Game
        # initialize the sys_player and the env_
        # [init_sys_Pos_value, init_env_Pos_value, 1]
        init_state = []
        rand_init_state = np.random.choice(init_state_list)

        if sys_init is None:
            init_state.insert(0, rand_init_state.x)
        else:
            init_state.insert(0, sys_init)
        if env_init is None:
            init_state.insert(1, rand_init_state.y)
        else:
            init_state.insert(1, env_init)

        # you always start in the state that belongs to system the player
        init_state.insert(2, 1)

        n = int(env.env.env_data["env_size"]["n"])
        m = int(env.env.env_data["env_size"]["m"])

        # plot the basic gridworld first
        self.plot_gridworld(env, plot_title)
        # A function to visualize the learned for the both the robot
        # for x in range(n):
        #     for y in range(m):
        time_step = 0
        # current state is of the form (Pos x Pos x 1)
        currentState = init_state
        game.restart(sys_robot=currentState[0], env_robot=currentState[1])

        # write a helper code to convert Pos value to (x, y) coordinates
        curr_sys_xy, curr_env_xy = self._convert_pos_to_xy(game.currentPosition, n, m)

        # need to offset the location of plotting the patch to match the cell it is in
        def offset_ticks(x, offset=0.5):
            return x + offset

        # plot this position on the gridworld
        sys_patch = self._add_patch(shape='circle', xy=tuple(map(offset_ticks, curr_sys_xy)))
        env_patch = self._add_patch(shape='circle', xy=tuple(map(offset_ticks, curr_env_xy)), color='blue')
        sys_reward = 0
        # add a text counter to the figure
        counter = self.ax.text(0.7, 1, f"system reward: {sys_reward}", fontsize=14, transform=self.ax.transAxes)
        # save the initial figure to plot
        plt.savefig(f"frames/grid_{time_step}.png", dpi=200)
        counter.remove()
        while time_step < ep_len:
            # save the figure at each time step and then use that to make a gif.
            sys_patch.remove()
            # increment the counter by 1
            time_step += 1

            # get the best action for the current state and transit accordingly
            best_sys_action = sys_player.chooseAction(player=0, state=game.currentPosition)

            # evolve according to the system action
            sys_reward += game.play_one_player_at_a_time(0, best_sys_action)

            # convert the  newState into (x, y) format
            curr_sys_xy, _ = self._convert_pos_to_xy(game.currentPosition, n, m)
            # update counter variable
            counter = self.ax.text(0.7, 1, f"system reward: {sys_reward}", fontsize=14, transform=self.ax.transAxes)
            # plot the action taken
            sys_patch = self._add_patch(shape='circle', xy=tuple(map(offset_ticks,curr_sys_xy)))
            plt.savefig(f"frames/grid_{time_step}_1.png", dpi=200)
            counter.remove()

            # get env's response
            best_evn_action = env_player.chooseAction(player=1, state=game.currentPosition)

            # evolve according to the env action
            game.play_one_player_at_a_time(1, best_evn_action)
            # convert the new state into (x, y) format
            _, curr_env_xy = self._convert_pos_to_xy(game.currentPosition, n, m)

            # remove the old sys robot patch
            env_patch.remove()

            # plot the action take by the env robot
            env_patch = self._add_patch(shape='circle', xy=tuple(map(offset_ticks, curr_env_xy)), color='blue')
            # update the counter variable
            counter = self.ax.text(0.7, 1, f"system reward: {sys_reward}", fontsize=14, transform=self.ax.transAxes)
            plt.draw()
            plt.pause(0.5)
            plt.savefig(f"frames/grid_{time_step}_2.png", dpi=200)
            counter.remove()

        plt.close(self.fig)
        self.make_gif('./frames/', f'./figures_and_gifs/N_{n}.gif')

    def make_gif(self, input_folder, save_filepath):
        episode_frames = []
        time_per_step = 0.5
        for root, _, files in os.walk(input_folder):
            file_paths = [os.path.join(root, file) for file in files]
            # sorted by modified time
            file_paths = sorted(file_paths, key=lambda x: os.path.getmtime(x))
            episode_frames = [imageio.imread(file_path)
                              for file_path in file_paths if file_path.endswith('.png')]
        episode_frames = np.array(episode_frames)
        imageio.mimsave(save_filepath, episode_frames, duration=time_per_step)


class dump_and_load:
    def __init__(self, file_name, path):
        # self.fileDir = os.path.dirname(os.path.realpath('__file__'))
        # sys.path.append(self.fileDir + path)
        self.file_name = file_name
        self.path = path
        # for accessing the file in a folder contained in the current folder
        self._complete_path = path + file_name

    def dump_file(self, object):
        # outfile = open(self._complete_path, 'w+b')
        with open(self._complete_path, 'w+b') as outfile:
            pickle.dump(object, outfile)
            outfile.close()

    def load_file(self):
        # print(self._complete_path)
        # infile = open(self._complete_path, 'rb')
        with open(self._complete_path, 'rb') as infile:
            unpickled_obj = pickle.load(infile)
            infile.close()

        return unpickled_obj


def construct_game(grid_size):
    # run the main function to build all the stuff in there
    TwoPlayerGame.TwoPlayerGame.dump_yaml_file('two_player_game/config/env.yaml', n=grid_size, m=grid_size)
    game = TwoPlayerGame.main()
    return game

def extract_permissive_str(grid_size):
    print("*********************Extracting (maximally) permissive strategy********************* \n")
    # if slugsfile_path is 3:
    slugsfile_path = f"two_player_game/slugs_file/CoRL_{grid_size}"
    str = extractExplicitPermissiveStrategy.PermissiveStrategy(slugsfile_path, run_local=False)
    str.convert_to_slugsin()
    str.convert_slugsin_to_permissive_str()
    env_str = None
    system_str = str.interpret_strategy_output(f"/slugs_file/CoRL_{grid_size}.txt")
    if print_child_node_length:
        for k, v in system_str.items():
            print(k , len(v['child_nodes']))
    if plot_graphs:
        for i in range(50):
            file_name = f"two_player_game/graphs/state{i}.dot"
            str.plot_graphs_from_dot_files(file_name, view=True)

    print("*********************Done Extracting strategy from Slugs tool********************* \n")
    return system_str, env_str

def create_xy_pos_mapping(x,y, nx, ny):
    # a method to map a respective x,y value on the grid to Pos
    # create a mapping from x,y grid position to Pos variable number : x_length*x + y
    Pos = nx*y + x
    return Pos

# write a code that does post processing on the strategy we get from slugs to remove the following:
# 1. The robot should not make any jumps. Although this does not appear in the str, its a safety measure to check for it
# 2. The sys robot can control only the x variable of the position tuple (x, y ,t)
# 3. The env robot can control only the y variable of the position tuple (x, y, t)

def pre_processing_of_str(strategy, nx, ny):
    # strategy is a dict [State Num], attributes [state_y_map], [stare_pos_map], [child_nodes]
    # robot should not make in jumps
    print("processing the raw strategy we get from slugs to remove undesired behavior \n")
    invalid_states = []
    for k, v in strategy.items():
        pos_tuple = v['state_pos_map']
        xy_tuple = v['state_xy_map']
        xy_sys = xy_tuple[0]
        xy_env = xy_tuple[1]
        # extract x, y, t
        x = pos_tuple[0]
        y = pos_tuple[1]
        t = pos_tuple[2]

        # the first pass is a sanity check to see if its a valid state or not.
        # Remove the state itself if it lies beyond the girdworld. Slugs in general does satisfy the constraint of NOT
        # transiting to states that lie beyond the world but the permissive strategy extension still compute trasitions
        # from states/(x,y)  positions beyond the gridworld to other state. This happens because, slugs stores the
        # variable into bit variable encoding which unfortunately gives rise extra values like [_,_] bit vectoring can
        # express range up to 4 and hence it goes upto 0 ...4 but the constraint in the transitions restricts it to
        # transit beyond the gridworld but not the other way around.

        # default del_flag to false:
        del_flag = False

        # So, remove states that do not belong in the gridworld of size (n,m)
        if xy_sys[0] >= nx or xy_sys[1] >= ny:
            del_flag = True
        if xy_env[0] >= nx or xy_env[1] >= ny:
            del_flag = True

        if del_flag:
            # safety try and except mechanism
            try:
                invalid_states.append(k)
                continue
            except KeyError:
                warnings.warn(f"State {k} not found in the strategy. This error should not occur")

        # go through the child nodes
        # list to keep track of invalid transitions:
        invalid_trans = []
        for node in v['child_nodes']:
            # there are children nodes to transit to. an empty children node list is give as [""]
            if node is not "":
            # get the pos tuple of the child node
                target_tuple = strategy[f"State {node}"]['state_pos_map']
                target_x = target_tuple[0]
                target_y = target_tuple[1]
                target_t = target_tuple[2]

                # a player can only transit to states that belong to him
                if t != target_t:
                    invalid_trans.append(node)
                    continue
                # check that the sys robot can only control the x variable
                if int(t) == 1:
                    # y should not change as the sys robot cannot control that variable
                    if y != target_y:
                        # v['child_nodes'].remove(node)
                        invalid_trans.append(node)
                        continue
                    if not (x + 1 == target_x or x - 1 == target_x or x + nx == target_x
                            or x - nx == target_x or x == target_x):
                        # its not a valid transition and hence should be removed from the strategy
                        # v['child_nodes'].remove(node)
                        invalid_trans.append(node)
                        continue
                # check that the env robot can only control the y variable
                if int(t) == 0:
                    # x should not change as the env robot cannot control that variable
                    if x != target_x:
                        # v['child_nodes'].remove(node)
                        invalid_trans.append(node)
                        continue
                    if not (y + 1 == target_y or y - 1 == target_y or y + nx == target_y or y - nx == target_y):
                        # v['child_nodes'].remove(node)
                        invalid_trans.append(node)
                        continue

        # remove all the nodes from the invalid transitions list L
        for ele in invalid_trans:
            v['child_nodes'].remove(ele)

    print(f"Total states pruned as they lied outside the gridworld are {len(invalid_states)}.\n "
          f"The current number of valid states for both players in strategy after pruning is {len(strategy) - len(invalid_states)} \n")
    for s in invalid_states:
        del strategy[s]


def _determine_action(game, curr_state, next_state):
    # extract all attributes of the current state
    curr_x = curr_state[0]
    curr_y = curr_state[1]
    curr_t = int(curr_state[2])

    # extract all attributes of the next state
    next_x = next_state[0]
    next_y = next_state[1]
    next_t = int(next_state[2])

    # get all the available action
    # fixed set of action : [up_s, down_s, left_s, right_s, stay_s]
    sys_actions = game.get_controlled_actions()
    # fixed set of actions :[up_e, down_e, left_e, right_e]
    env_actions = game.get_uncontrolled_actions()
    nx = game.env.env_data["env_size"]["n"]

    if next_x-curr_x > 0:
        if next_x - curr_x == 1:
            return sys_actions[3]
        if next_x - curr_x == nx:
            return sys_actions[0]
    elif next_x - curr_x < 0:
        if next_x - curr_x == -1:
            return sys_actions[2]
        if next_x - curr_x == -nx:
            return sys_actions[1]
    elif next_y - curr_y > 0:
        if next_y - curr_y == 1:
            return env_actions[3]
        if next_y - curr_y == nx:
            return env_actions[0]
    elif next_y - curr_y < 0:
        if next_y - curr_y == -1:
            return env_actions[2]
        if next_y - curr_y == -nx:
            return env_actions[1]
    elif curr_x == next_x:
        return sys_actions[4]
    else:
        print("********************invalid transition**********************")
        return 0


def update_transition_function(game, strategy):
    for k, v in strategy.items():
        # get the state number from the strategy corresponding to k (State n)
        curr_game_state = v['state_pos_map']
        # determine the action corresponding to the current transition
        for child_state_number in v['child_nodes']:
            # convert State n to (x, y, t) format
            next_game_state = strategy[f"State {child_state_number}"]["state_pos_map"]
            action = _determine_action(game, curr_game_state, next_game_state)
            # get the corresponding game state
            curr_G_state = game.S[curr_game_state[0]][curr_game_state[1]][int(curr_game_state[2])]
            next_G_state = game.S[next_game_state[0]][next_game_state[1]][int(next_game_state[2])]
            game.set_transition_function(curr_state=curr_G_state, action=action, next_state=next_G_state)


def create_G_hat(strategy, game):
    # give a game G and strategy update the following:
    # 1. Remove all the State from which there does not exists any transition (these are sink/ ghost states)
    # 2. Remove all invalid states for System player
    # 3. Remove all invalid states for env player
    # 4. Remove all invalid initial states
    # 5. Update Transition function
    # 6. Update W = (L, True)
    print("Creating G_hat game \n")
    # get pos_x and pos_y values
    pos_x_length = game.env.pos_x
    pos_y_length = game.env.pos_y

    # copy the game in
    game_G_hat = copy.deepcopy(game)

    # 1-4. Remove all the States that do no appear in the permissive strategy for various reasons
    # list to keep track of states to remove
    invalid_states = []
    # iterate over all the possible states and check that state does exist in the permissive
    for ix in pos_x_length:
        for iy in pos_y_length:
            for p in range(0, 2):
                # if this state does on exist in the strategy then remove it
                exist_flag = False
                for k, v in strategy.items():
                    if v['state_pos_map'] == (ix, iy, str(p)):
                        exist_flag = True
                        # print(f"Removing state {ix, iy, p}")
                # after searching through all the whole strategy remove the state if exist_flag is False
                if not exist_flag:
                    print(f"Removing state {ix, iy, p}")
                    invalid_states.append((ix, iy, p))

    # remove duplicates from the list
    invalid_states = list(dict.fromkeys(invalid_states))

    # remove all the states not present in the strategy and also update system and env states list
    for r_x, r_y, r_t in invalid_states:
        # remove elements from the system state space
        if r_t == 1:
            game_G_hat.S_s.remove(game_G_hat.S[r_x][r_y][r_t])
            # update the list of initial states to be states where both the sys robot and env robot
            # cannot be in the same location

            # as the states are shallow copied, removing objects form the sys_states list removes from the initial
            # states list as well
            # game_G_hat.Init.remove(game_G_hat.S[r_x][r_y][r_t])

        # remove elements form the env state space
        else:
            game_G_hat.S_e.remove(game_G_hat.S[r_x][r_y][r_t])
        # remove elements from the parent State space list
        game_G_hat.S[r_x][r_y][r_t] = None

    # 5. Update the transition function
    update_transition_function(game_G_hat, strategy)

    # 6. Update W
    game_G_hat.set_W(phi=True)

    print("Done Constructing G_hat game")
    return game_G_hat

@deprecated
def testGame(sys_player, env_player, game, iterations, episode_len=1000):
    reward_per_episode = []

    for episode in range(iterations):
        cumulative_reward = 0
        # print after 10 % iteration being completed
        if (episode % (iterations /10) == 0):
            print("%d%%" % (episode * 100 / iterations))
        game.restart()
        step = 0
        while step < episode_len:
            step += 1
            # get the initial state
            currentState = game.currentPosition

            # get the action for the system player
            sys_action = sys_player.chooseAction(player=0, state=currentState)
            # evolve on the game according to system's play
            game.play_one_player_at_a_time(0, sys_action)

            # now env player chooses action from the new intermediate state
            intermediateState = game.currentPosition
            env_action = env_player.chooseAction(player=1, state=intermediateState)
            # evolve on the game
            # currentState = game.currentPosition  # get the new updated position
            reward = game.play_one_player_at_a_time(1, env_action)

            # move according to the actions picked and get the reward
            # reward = game.play(sys_action, env_action)

            # update cummulatve reward value
            cumulative_reward += reward

            # get the next state
            nextState = game.currentPosition

            # update Q and V values
            sys_player.learn(currentState, nextState, actions=[sys_action, env_action], reward=reward)
            env_player.learn(currentState, nextState, actions=[sys_action, env_action], reward=-reward)

    return reward_per_episode

def test_q_learn_alternating_markov_game(sys_player, env_player, game, iterations, episode_len=1000, tol=1e-10):
    # tolerance has to be this high to ensure that state values converge to the ideal value
    sys_reward_per_episode = []
    env_reward_per_episode = []
    delta = []

    # get the previous step V for system robot
    old_sys_V = sys_player.get_V()

    for episode in range(int(iterations)):
        # increment episode number
        episode += 1
        # compute max V change after every 10**4 iterations
        if episode % 10**3 == 0:

            max_diff = max(abs(old_sys_V - new_sys_V))
            delta.append(max_diff)

            if max_diff < tol:
                print(f"Breaking at iteration : {episode}. Thw current current delta V = {max_diff}")
                break

            # update the old value with the current value
            old_sys_V = new_sys_V

        sys_cumulative_reward = 0
        env_cumulative_reward = 0
        # print after 10 % iteration being completed
        if (episode % (iterations / 100) == 0):
            print("%d%%" % (episode * 100 / iterations))
        game.restart()
        step = 0

        while step < episode_len:
            step += 1

            # get the initial state
            oldState = game.currentPosition
            # print(oldState)
            # get the system action
            sys_action = sys_player.chooseAction(player=0, state=oldState)

            # evolve according to the system action and get the reward
            sys_reward = game.play_one_player_at_a_time(player=0, action=sys_action)

            #  system learns
            sys_player.learn(oldState, game.currentPosition, sys_action, sys_reward)

            # update currentState in the gridworld
            oldState = game.currentPosition

            # get the env action
            env_action = env_player.chooseAction(player=1, state=oldState)

            # evolve according to the env action and get the reward
            env_reward = game.play_one_player_at_a_time(player=1, action=env_action)

            # env learns
            env_player.learn(oldState, game.currentPosition, env_action, -1*env_reward)

            # update the cumulative reward
            sys_cumulative_reward += sys_reward
            env_cumulative_reward += env_reward

        # check if V for the system converged or now
        new_sys_V = sys_player.get_V()
        sys_reward_per_episode.append(sys_cumulative_reward)
        env_reward_per_episode.append(env_cumulative_reward)

    return sys_reward_per_episode, env_reward_per_episode, delta, episode


def compute_optimal_strategy_using_rl(sys_str, game_G_hat, saved_flag, iterations):
    # if the saved_flag is True then directly load binary filed from the 'saved_players' directory and use that to plot
    # policy
    n = game_G_hat.env.env_data["env_size"]["n"]
    reward_per_episode = [None]
    rl_time = None
    # create instances to dump and load file
    system_robot_dump = dump_and_load(file_name=f'sys_agent_N_{n}', path='saved_players/')
    env_robot_dump = dump_and_load(file_name=f'env_agent_N_{n}', path='saved_players/')
    rl_env_dump = dump_and_load(file_name=f'rl_gridworld_N_{n}', path='saved_players/')
    reward_dump = dump_and_load(file_name=f'reward_per_ep_N_{n}', path='saved_players/')

    if not saved_flag:
        print("*********************Starting learning routine to learn the optimal strategy*********************")
        # define some initial values
        decay = 10**(-2. / iterations * 0.05)
        height = game_G_hat.env.pos_x.stop
        width = game_G_hat.env.pos_y.stop

        # randomly choose an initial state
        rand_init_state = np.random.choice(game_G_hat.Init)

        # create agents
        # sys_agent = minimax_agent(game_G_hat.T, decay=decay)
        # env_agent = rand_agent(game_G_hat.T)
        sys_agent = q_agent(game_G_hat.T, player=0, decay=decay, epsilon=0.1)
        env_agent = q_agent(game_G_hat.T, player=1, decay=decay)

        # create the gridworld
        rl_game = Gridworld(height, width, rand_init_state.x, rand_init_state.y, game_G_hat.Init)

        # call a function that runs tests for you
        # reward_per_episode = testGame(sys_agent, env_agent, game, iterations=10**5) # deprecated
        start = time.clock()
        reward_per_episode = test_q_learn_alternating_markov_game(sys_agent, env_agent,
                                                                  game=rl_game,
                                                                  iterations=iterations,
                                                                  episode_len=30)
        stop = time.clock()
        rl_time = stop - start
        print(f"Took {rl_time} s to compute the optimal strategy using q-learning for the controlled agent")
        # dump the context
        reward_dump.dump_file(reward_per_episode)
        # display result
        del_V_plot = plotterClass(fig_title="max del_V_per_10k", ylabel="max $\Delta V$/$10^{3}$ ",
                                  xlabel="Iterations (x $10^{3}$)")
        del_V_plot.ax.plot(reward_per_episode[2], label="max V change")
        del_V_plot.ax.set_yscale('log')
        del_V_plot.ax.legend()
        # save both the plots in two_players_game/plot local directory
        del_V_plot.save_fig(f"two_player_game/plots/{del_V_plot.fig_title}_N_{n}.png")
        # pause for 0.5 seconds for visualization and then close the figure
        plt.pause(0.5)
        plt.close()
        # compute desired state values to plot against on the y-axis
        desired_values = get_desired_v_value(k=game_G_hat.env.env_data["env_size"]["n"] - 1)

        # plot policy of sys_player
        V_plot = plotterClass(fig_title="State Value", ylabel="State Value", xlabel="$s_{i} \in \^S_{s}$")
        plot_state_value(sys_str, sys_agent.V, desired_values, V_plot.ax)
        V_plot.save_fig(f"two_player_game/plots/{V_plot.fig_title}_N_{n}.png")
        # pause for 0.5 seconds for visualization and then close the figure
        plt.pause(0.5)
        plt.close()

        # dump instances of the player classes - sys_player & env player and instance of the rl_game
        env_robot_dump.dump_file(env_agent)
        system_robot_dump.dump_file(sys_agent)
        rl_env_dump.dump_file(rl_game)

    # load the binary pickled file
    unpickled_sys = system_robot_dump.load_file()
    unpickled_env = env_robot_dump.load_file()
    unpickled_rl_game = rl_env_dump.load_file()

    # lets do some fancy visualization stuff here
    print("Plotting the policy learned")
    play_game(env=game_G_hat, init_state_list=game_G_hat.Init, sys_player=unpickled_sys, env_player=unpickled_env,
              game=unpickled_rl_game)
    print("Done animating")

    # plot the game policy learned
    plot_pi = plotterClass(fig_title=f"Policy for G_hat_N_{n}", xlabel='$pos_x$', ylabel='$pos_y$')
    plot_pi.plot_policy(env=game_G_hat, sys_player=unpickled_sys, sys_str=sys_str)
    plot_pi.save_fig(f"two_player_game/plots/{plot_pi.fig_title}_N_{n}.png", dpi=500)
    plt.pause(1)
    plt.close()

    # plot state value on the game G_hat
    plot_v_opt = plotterClass(fig_title=f"State_Values_game_world_N_{n}", xlabel="$pos_x$", ylabel="$pos_y$")
    plot_v_opt.plot_state_value(env=game_G_hat, sys_player=unpickled_sys)
    plot_v_opt.save_fig(f"two_player_game/plots/{plot_v_opt.fig_title}.png", dpi=500)
    plt.pause(1)
    plt.close()

    # plot state value on the game G_hat
    plot_q_opt = plotterClass(fig_title=f"Action_Values_game_world_N_{n}", xlabel="$pos_x$", ylabel="$pos_y$")
    plot_q_opt.plot_q_value(env=game_G_hat, sys_player=unpickled_sys)
    plot_q_opt.save_fig(f"two_player_game/plots/{plot_q_opt.fig_title}.png", dpi=500)
    plt.pause(1)
    plt.close()

    return rl_time, reward_per_episode[-1]

def processing_w_obstacles(strategy):
    """
    Although we do not transit to the obstacle state, permissive stratetegy does into consideration starting from these
    states. So we need to prune those states from the strategy which belong to the obstacle
    :return:
    :rtype:
    """
    # cell that belong to the obstacle
    obs_cell_lst = [2, 3, 4, 9, 31, 36, 37, 38]

    # a list to keep track of the invalid
    invalid_states = []
    for k, v in strategy.items():
        # get the state number from the strategy corresponding to k (State n)
        curr_game_state = v['state_pos_map']
        x = curr_game_state[0]
        y = curr_game_state[1]
        # (x, y, t) if x or u belong to obs_cell_lst then remove that state from the strategy
        if x in obs_cell_lst:
            invalid_states.append(k)
        if y in obs_cell_lst:
            invalid_states.append(k)

    # remove redundant elements from the list
    invalid_states = list(dict.fromkeys(invalid_states))

    for ele in invalid_states:
        del strategy[ele]

def play_game(env, init_state_list, sys_player, env_player, game):

    # create plotter object
    final_plot = plotterClass(fig_title="Learned policy visualization", xlabel='x', ylabel='y')
    final_plot.roll_out_final_policy(env, init_state_list, sys_player, env_player,
                              game, ep_len=30)
    plt.show()


def get_desired_v_value(k):
    """
    The Reward function used in the paper is a sum of discounted reward of the sum(k, gamma.k * reward) which
    converges to the formula gamma**k/ (1 - gamme) where k = N - 1.
    :param k:
    :type k: int
    :return: a list of the desired state values that the system agent should converge to
    :rtype: list
    """

    def reward_func(k, gamma=0.9):
        return gamma ** k / (1 - gamma)

    k_vals = [i for i in range(k+1)]
    desired_values = list(map(reward_func, k_vals))

    return desired_values

# helper method to plot the scatter plot of state and its state value
def plot_state_value(sys_str, V, desired_V, ax):

    state_num_list = []
    value_list = []
    # process the state value for system player only
    for state, state_value in V.items():
        # get the respective state number for a  pos value state (Pos x Pos x t)

        # find the matching State
        for k, v in sys_str.items():
            # add only those that belong to the system player
            if state[2] == 1 and v['state_pos_map'] == (state[0], state[1], str(state[2])):
                num = re.findall(r'\d+', k)
                state_num_list.append(num)
                value_list.append(state_value)

    x =[i for i in range(len(value_list))]
    ax.scatter(x, value_list, marker='*', c="blue", label='state')
    ax.grid(True, axis='y')
    ax.legend()
    plt.yticks(desired_V)


def update_strategy_w_state_pos_map(strategy, x_length, y_length):
    # create a mapping from th state we get form the strategy and the state we have in our code

    # create a tmp_state_tuple
    for k, v in strategy.items():
        xy_sys = v['state_xy_map'][0]
        xy_env = v['state_xy_map'][1]
        pos_sys = create_xy_pos_mapping(xy_sys[0], xy_sys[1], x_length, y_length)
        pos_env = create_xy_pos_mapping(xy_env[0], xy_env[1], x_length, y_length)
        # t = 1
        t = v['player']
        # print(f"{k} : {xy_sys}, {xy_env}0 : ({pos_env}, {pos_env}, {t})")

        # update the state_pos mapping variable in sys_str
        strategy[k]['state_pos_map'] = (pos_sys, pos_env, t)
        if print_state_pos_map:
            print(k, strategy[k]['state_pos_map'])


def parse_addition_args(args):
    """
    A method to parse the additional arguements passed by the user
    1. save_flag = a flag to use the saved player from the src/saved_players directory
    2. gridsize = integer number depiciting which gridsize to load - 3,4,5,6 load normal grid env while 7 loads the
    fancy obstacle env
    :param args: list
    :type args:
    :return:
    :rtype:
    """
    if len(args) > 1:
        script_name = args[0]
        if args[1] == 'False':
            save_flag = False
        else:
            save_flag = True
        grid_size = int(args[2])
    else:
        print("Uusin default argument values : N = 3 ; saved_player = False(will train from scratch)")
        save_flag = True
        grid_size = 3
    return save_flag, grid_size


def dump_to_yaml_file(grid_size, slugs_time, rl_time, iterations, total_states, sys_states):

    data = dict(
        grid_size=grid_size,
        slugs_time=float(slugs_time),
        rl_time=rl_time,
        iterations=iterations,
        total_states=total_states,
        sys_states=sys_states
    )

    with open(r'results/log.yaml', 'a') as file:
        yaml.dump(data, file)


def print_code_logo():
    print("********     ********     ****       ** \n\
********     ********     ******     ** \n\
***          **    **     **   **    ** \n\
***          **    **     **  **     ** \n\
***          **    **     ** **      ** \n\
***          **    **     ***        ** \n\
***          **    **     ****       ** \n\
********     ********     ** **      ******* \n\
********     ********     **  **     *******")

if __name__ == "__main__":

    additional_args = sys.argv
    save_flag, grid_size = parse_addition_args(additional_args)
    print("************************************************")
    print_code_logo()
    print("************************************************")
    time.sleep(2)
    print("*********************Constructing games G*********************")
    game = construct_game(grid_size)
    time.sleep(1)
    x_length = game.env.env_data["env_size"]["n"]
    y_length = game.env.env_data["env_size"]["m"]
    start = time.clock()
    sys_str, env_str = extract_permissive_str(grid_size=grid_size)
    stop = time.clock()
    slugs_time = stop - start
    print(f"Took {slugs_time} s to compute the (maximally) permissive strategy from slugs tool")
    time.sleep(2)
    a_c = game.get_controlled_actions()
    # create a mapping from th state we get form the strategy and the state we have in our code
    update_strategy_w_state_pos_map(sys_str, x_length, y_length)

    # do some pre-processing on the strategy we get
    print("*********************Processing strategy we get from slugs to remove invalid "
          "transitions*********************\n")
    pre_processing_of_str(sys_str, x_length, y_length)

    if grid_size == 7:
        processing_w_obstacles(sys_str)
    if print_str_after_processing:
        print("*********************Strategy after removing all the invalid transitions*********************")
        for k, v in sys_str.items():
            print(f"{k}: {v}")
    time.sleep(2)
    print("*********************Constructing game G_hat*********************")
    # update Game G to get G_hat
    game_G_hat = create_G_hat(strategy=sys_str, game=game)

    # print the updated transition matrix
    if print_updated_transition_function:
        print("The Updated transition matrix is: ")
        game_G_hat.print_transition_matrix()

    time.sleep(2)
    print("*********************Computing the optimal strategy*********************")
    rl_time, iters = compute_optimal_strategy_using_rl(sys_str, game_G_hat, iterations=10*10**5, saved_flag=save_flag)

    # after the computations is done dump the result to a yaml file
    dump_to_yaml_file(grid_size=grid_size, slugs_time=slugs_time, rl_time=rl_time, iterations=iters,
                      total_states=game_G_hat.get_total_state_count(),
                      sys_states=game_G_hat.get_total_sys_state_count())
