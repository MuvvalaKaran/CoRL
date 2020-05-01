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


from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from src.learning_reward_function.Players.minimaxq_player import MinimaxQPlayer as minimax_agent
from src.learning_reward_function.Players.random_player import RandomPLayer as rand_agent
from src.learning_reward_function.Players.qplayer import QPlayer as q_agent
from src.learning_reward_function.MakovGame.diagonal_players import Gridworld
from src.two_player_game import TwoPlayerGame
from src.two_player_game import extractExplicitPermissiveStrategy

# flag to plot BDD diagram we get from the slugs file
plot_graphs = False
# flag to print length of child nodes
print_child_node_length = False
# flag to print strategy after preprocessing
print_str_after_processing = True
# flag to print state num and state_pos_map value
print_state_pos_map = True
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

    def __init__(self, fig_title, robo_file_path=None):
        self.fig = None
        self.ax = None
        self.fig_title = fig_title

        # initialize a figure an axes handle
        self._create_ax_fig()
        if robo_file_path is not None:
            self.robot_patch = mping.imread(robo_file_path)

    def _create_ax_fig(self):
        self.fig = pl.figure(num=self.fig_title)
        self.ax = self.fig.add_subplot(111)

    def plot_gridworld_with_label(self):
        raise NotImplementedError

    def plot_policy(self):
        raise NotImplementedError

    def gridworld(self, env, print_grid_num=False, *args):
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
                    self.plot_grid_num((off_x, off_y), value=f"{x, y}")

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
        x, y, = xy

        self.ax.annotate(value, xy=(x, y), xycoords='data',
                    horizontalalignment='center',
                    verticalalignment='center',
                    **kwargs)

    def _add_patch(self, shape, xy, color='green', robo_image=None):
        x, y = xy
        shape_case = {
            'circle': 1,
            'rect': 2,
            'arrow': 3,
            'robot': 4
        }

        if shape_case[shape] == 1:
            self.ax.add_patch(patches.Circle(
                (x, y),
                radius=0.2,
                color=color,
                alpha=1  # transparency value
            ))
        elif shape_case[shape] == 2:
            self.ax.add_patch(patches.Rectangle(
                (x, y),
                0.1,
                0.1,
                facecolor=color,
                edgecolor='black',
                alpha=0.5 # transparency value
            ))
        elif shape_case[shape] == 3:
            pass
        elif shape_case[shape] == 4:
            imageBox = OffsetImage(robo_image, zoom=0.1)
            ab = AnnotationBbox(imageBox, xy, frameon=False)
            self.ax.add_artist(ab)
        else:
            return warnings.warn("Not a valid shape. Need to add that patch to the _add_patch function")

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
        self.gridworld(env, plot_title)
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
        self._add_patch(shape='circle', xy=tuple(map(offset_ticks, curr_sys_xy)))
        self._add_patch(shape='circle', xy=tuple(map(offset_ticks, curr_env_xy)), color='blue')
        sys_reward = 0
        # add a text counter to the figure

        self.ax.text(0.7, 1, f"system reward: {sys_reward}", fontsize=14, transform=self.ax.transAxes)

        plt.savefig(f"frames/grid_{time_step}.png", dpi=200)
        self.ax.clear()
        self.gridworld(env, plot_title)
        while time_step < ep_len:
            # save the figure at each time step and then use that to make a gif.

            # increment the counter by 1
            time_step += 1

            # get the best action for the current state and transit accordingly
            best_sys_action = sys_player.chooseAction(player=0, state=game.currentPosition)

            # evolve according to the system action
            sys_reward += game.play_one_player_at_a_time(0, best_sys_action)
            # newState = currentState[0] + switch_case[best_sys_action]
            # currentState = newState

            # convert the  newState into (x, y) format
            curr_sys_xy, _ = self._convert_pos_to_xy(game.currentPosition, n, m)

            # plot the action taken
            self._add_patch(shape='circle', xy=tuple(map(offset_ticks,curr_sys_xy)))

            # plt.savefig(f"frames/grid_{time_step}_1.png", dpi=200)
            # get env's response
            best_evn_action = env_player.chooseAction(player=1, state=game.currentPosition)

            # evolve according to the env action
            # newState = currentState[1] + switch_case[best_evn_action]
            game.play_one_player_at_a_time(1, best_evn_action)
            # convert the new state into (x, y) format
            _, curr_env_xy = self._convert_pos_to_xy(game.currentPosition, n, m)

            # plot the action take by the env robot
            self._add_patch(shape='circle', xy=tuple(map(offset_ticks, curr_env_xy)), color='blue')

            self.ax.text(0.7, 1, f"system reward: {sys_reward}", fontsize=14, transform=self.ax.transAxes)
            plt.draw()
            plt.pause(0.5)
            plt.savefig(f"frames/grid_{time_step}.png", dpi=200)
            # clear the give screen
            self.ax.clear()

            # plot a fresh figure after each player has played his turn
            self.gridworld(env, plot_title)


        plt.close(self.fig)
        # plt.show()
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
        self.file_name = file_name
        self.path = path
        self._complete_path = path + file_name

    def dump_file(self, object):
        outfile = open(self._complete_path, 'w+b')
        pickle.dump(object, outfile)
        outfile.close()

    def load_file(self):
        infile = open(self._complete_path, 'rb')
        unpickled_obj = pickle.load(infile)
        infile.close()

        return unpickled_obj



def construct_game():
    print("Constructing game G \n")
    # run the main function to build all the stuff in there
    game = TwoPlayerGame.main()
    return game

def extract_permissive_str(slugsfile_path=None):
    print("Extracting (maximally) permissive strategy \n")
    if slugsfile_path is None:
        slugsfile_path = "two_player_game/slugs_file/CoRL_1"
    str = extractExplicitPermissiveStrategy.PermissiveStrategy(slugsfile_path, run_local=False)
    str.convert_to_slugsin()
    str.convert_slugsin_to_permissive_str()
    env_str = None
    system_str = str.interpret_strategy_output("/slugs_file/CoRL_1.txt")
    if print_child_node_length:
        for k, v in system_str.items():
            print(k , len(v['child_nodes']))
    if plot_graphs:
        for i in range(50):
            file_name = f"two_player_game/graphs/state{i}.dot"
            str.plot_graphs_from_dot_files(file_name, view=True)

    return system_str, env_str

def create_xy_pos_mapping(x,y, nx, ny):
    # a method to map a respective x,y value on the grid to Pos
    # create a mapping from x,y grid position to Pos variable number : x_length*x + y
    Pos = nx*y + x
    return Pos

def convert_pos_xy_mapping(pos_x, pos_y, nx, ny):
    raise NotImplementedError

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
            # if state belongs to sys then only the x variable can change

            # compare x and y for any jump and if any exists, remove them
            # if not (x + 1 == target_x or x - 1 == target_x or x + nx == target_x or x - nx == target_x):
            #     # you can stay in the same cell if you are the sys player
            #     if int(t) == 1 and x == target_x:
            #         continue
            #     else:
            #         # its not a valid transition and hence should be removed from the strategy
            #         v['child_nodes'].remove(node)
            #         continue
            # if not (y + 1 == target_y or y - 1 == target_y or y + nx == target_y or y - nx == target_y):
            #     v['child_nodes'].remove(node)
            #     continue
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

    # 1-4. Remove all the States that do no appear in the permissive for various reasons
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

    for episode in range(iterations):
        # increment episode number
        episode += 1
        # compute max V change after every 10**4 iterations
        if episode % 10**4 == 0:

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
        if (episode % (iterations / 10) == 0):
            print("%d%%" % (episode * 100 / iterations))
        game.restart()
        step = 0

        while step < episode_len:
            step += 1

            # get the initial state
            oldState = game.currentPosition

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

    print(sys_player.V)
    return sys_reward_per_episode, env_reward_per_episode, delta


def compute_optimal_strategy_using_rl(sys_str, game_G_hat, iterations=10*10**5, saved_flag=True):
    # if the saved_flag is True then directly load binary filed from the 'saved_players' directory and use that to plot
    # policy
    n = game_G_hat.env.env_data["env_size"]["n"]

    # create instances to dump and load file
    system_robot_dump = dump_and_load(file_name=f'sys_agent_N_{n}', path='saved_players/')
    env_robot_dump = dump_and_load(file_name=f'env_agent_N_{n}', path='saved_players/')
    rl_env_dump = dump_and_load(file_name=f'rl_gridworld_N_{n}', path='saved_players/')

    if not saved_flag:
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
        # reward_per_episode = testGame(sys_agent, env_agent, game, iterations=10**5)
        reward_per_episode = test_q_learn_alternating_markov_game(sys_agent, env_agent,
                                                                  game=rl_game,
                                                                  iterations=10*10**5,
                                                                  episode_len=30)

        # display result
        # finc max of sys and env reward for plotting purposes
        # x1 = max(reward_per_episode[0])
        # x2 = max(reward_per_episode[1])
        #
        # plt.plot(reward_per_episode[0])
        # plt.plot(reward_per_episode[1])
        # plt.show()
        fig_title = "Rewards per episode for each player"
        fig = pl.figure(num=fig_title)
        ax1 = fig.add_subplot(111)

        # pl.hist(reward_per_episode[0], density=False, bins=30)
        # pl.hist(reward_per_episode[1], density=False, bins=30)
        ax1.plot(reward_per_episode[2], label="max V change")
        ax1.set_yscale('log')
        # ax.plot(reward_per_episode[1], label="env reward")
        ax1.legend()

        # compute desired state values
        desired_values = get_desired_v_value(k=game_G_hat.env.env_data["env_size"]["n"] - 1)
        # plot policy of sys_player
        fig = pl.figure(num="State Value")
        ax2 = fig.add_subplot(111)
        plot_state_value(sys_str, sys_agent.V, desired_values, ax2)

        # dump instances of the player classes - sys_player & env player and instance of the rl_game
        env_robot_dump.dump_file(env_agent)
        system_robot_dump.dump_file(sys_agent)
        rl_env_dump.dump_file(rl_game)

    # load the binary pickled file
    unpickled_sys = system_robot_dump.load_file()
    unpickled_env = env_robot_dump.load_file()
    unpickled_rl_game = rl_env_dump.load_file()
    # testing loading that player

    # plt.show()
    # plot policy

    # lets do some fancy visualization stuff here
    print("Plotting the policy learnt")
    play_game(env=game_G_hat, init_state_list=game_G_hat.Init, sys_player=unpickled_sys, env_player=unpickled_env,
              game=unpickled_rl_game)
    print("Done animating")

def play_game(env, init_state_list, sys_player, env_player, game):

    # create plotter object
    final_plot = plotterClass(fig_title="Learned policy visualization")
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
    ax.scatter(x, value_list, marker='*', c="blue")
    ax.grid(True, axis='y')
    plt.yticks(desired_V)


def updatestratetegy_w_state_pos_map(strategy, x_length, y_length):
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


def main():
    # construct_game()
    game = construct_game()
    x_length = game.env.env_data["env_size"]["n"]
    y_length = game.env.env_data["env_size"]["m"]
    sys_str, env_str = extract_permissive_str()
    a_c = game.get_controlled_actions()
    updatestratetegy_w_state_pos_map(sys_str)
    sys_srt = pre_processing_of_str(sys_str, x_length, y_length)
    if print_str_after_processing:
        print("Strategy after removing all the invalid transitions")
        for k, v in sys_str.items():
            print(f"{k}: {v}")

    # use these str to update the transition matrix
    # update_transition_function(game, sys_str)

    # update Game G to get G_hat
    game_G_hat = create_G_hat(strategy=sys_str, game=game)

    # print the updated transition matrix
    if print_updated_transition_function:
        print("The Updated transition matrix is: ")
        game_G_hat.print_transition_matrix()

    return game_G_hat

if __name__ == "__main__":
    # construct_game()
    game = construct_game()
    x_length = game.env.env_data["env_size"]["n"]
    y_length = game.env.env_data["env_size"]["m"]
    sys_str, env_str = extract_permissive_str()
    a_c = game.get_controlled_actions()
    # create a mapping from th state we get form the strategy and the state we have in our code

    # create a tmp_state_tuple
    for k, v in sys_str.items():
        xy_sys = v['state_xy_map'][0]
        xy_env = v['state_xy_map'][1]
        pos_sys = create_xy_pos_mapping(xy_sys[0], xy_sys[1], x_length, y_length)
        pos_env = create_xy_pos_mapping(xy_env[0], xy_env[1], x_length, y_length)
        # t = 1
        t = v['player']
        # print(f"{k} : {xy_sys}, {xy_env}0 : ({pos_env}, {pos_env}, {t})")

        # update the state_pos mapping variable in sys_str
        sys_str[k]['state_pos_map'] = (pos_sys, pos_env, t)
        if print_state_pos_map:
            print(k, sys_str[k]['state_pos_map'])

        # game.set_transition_function(game.S[pos_sys][pos_env][t], )
    # do some preprocessing on the straetegy we get
    sys_srt = pre_processing_of_str(sys_str, x_length, y_length)
    if print_str_after_processing:
        print("Strategy after removing all the invalid transitions")
        for k, v in sys_str.items():
            print(f"{k}: {v}")

    # use these str to update the transition matrix
    # update_transition_function(game, sys_str)

    # update Game G to get G_hat
    game_G_hat = create_G_hat(strategy=sys_str, game=game)

    # print the updated transition matrix
    if print_updated_transition_function:
        print("The Updated transition matrix is: ")
        game_G_hat.print_transition_matrix()

    compute_optimal_strategy_using_rl(sys_str, game_G_hat, saved_flag=True)

