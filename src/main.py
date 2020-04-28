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

def test_q_learn_alternating_markov_game(sys_player, env_player, game, iterations, episode_len=1000):
    sys_reward_per_episode = []
    env_reward_per_episode = []

    for episode in range(iterations):
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

        sys_reward_per_episode.append(sys_cumulative_reward)
        env_reward_per_episode.append(env_cumulative_reward)

    print(sys_player.V)
    return sys_reward_per_episode, env_reward_per_episode



def compute_optimal_strategy_using_rl(game_G_hat, iterations=10**5):
    # define some initial values
    decay = 10**(-2. / iterations * 0.05)
    height = game_G_hat.env.pos_x.stop
    width = game_G_hat.env.pos_y.stop

    # randomly choose an initial state
    rand_init_state = np.random.choice(game_G_hat.Init)

    # create agents
    # sys_agent = minimax_agent(game_G_hat.T, decay=decay)
    # env_agent = rand_agent(game_G_hat.T)
    sys_agent = q_agent(game_G_hat.T, player=0, decay=decay)
    env_agent = q_agent(game_G_hat.T, player=1, decay=decay)

    # create the gridworld
    game = Gridworld(height, width, rand_init_state.x, rand_init_state.y, game_G_hat.Init)

    # call a function that runs tests for you
    # reward_per_episode = testGame(sys_agent, env_agent, game, iterations=10**5)
    reward_per_episode = test_q_learn_alternating_markov_game(sys_agent, env_agent, game, iterations=4*10**5, episode_len=30)

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
    ax = fig.add_subplot(111)

    pl.hist(reward_per_episode[0], density=False, bins=30)
    pl.hist(reward_per_episode[1], density=False, bins=30)
    # ax.plot(reward_per_episode[0], label="sys reward")
    # ax.plot(reward_per_episode[1], label="env reward")
    # ax.legend()

    plt.show()

    # plot policy of sys_player

    # plot policy


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

    compute_optimal_strategy_using_rl(game_G_hat)

