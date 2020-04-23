# parent script file
# 1. Construct a game G and Extract a permissive str using slugs
# 2. Check if the specification is realizable and if so apply the str to G to compute G_hat
# 3. Compute the most optimal str that maximizes the reward while also satisfying the safety and liveness guarantees
# 4. Map the optimal str back to G
import time

from src.two_player_game import TwoPlayerGame
from src.two_player_game import extractExplicitPermissiveStrategy

# flag to plot BDD disgram we get from the slugs file
plot_graphs = False
# flag to print length of child nodes
print_child_node_length = False
# flag to print strategy after preprocessing
print_str_after_processing = False
# flag to print state num and state_pos_map value
print_state_pos_map = False
# flag to print the update transition matrix
print_updated_transition_function = True

def construct_game():
    # run the main function to build all the stuff in there
    game = TwoPlayerGame.main()
    # print(game.get_atomic_propositions())
    # print(game.get_initial_state())
    # print(game.get_players_data())
    return game

def extract_permissive_str():
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
    Pos = nx*x + y
    return Pos

# write a code that does post processing on the strategy we get from slugs to remove the following:
# 1. The robot should not make any jumps. Although this does not appear in the str, its a safety measure to check for it
# 2. The sys robot can control only the x variable of the position tuple (x, y ,t)
# 3. The env robot can control only the y variable of the position tuple (x, y, t)

def pre_processing_of_str(strategy, nx, ny):
    # strategy is a dict [State Num], attributes [state_y_map], [stare_pos_map], [child_nodes]
    # robot should not make in jumps
    for k, v in strategy.items():
        pos_tuple = v['state_pos_map']
        # extract x, y, t
        x = pos_tuple[0]
        y = pos_tuple[1]
        t = pos_tuple[2]

        # go through the child nodes
        # list to keep track of invalid transitions:
        invalid_trans = []
        for node in v['child_nodes']:
            # get the pos tuple of the child node
            target_tuple = strategy[f"State {node}"]['state_pos_map']
            target_x = target_tuple[0]
            target_y = target_tuple[1]

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

    for k,v in strategy.items():
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
    update_transition_function(game, sys_str)

    # print the updated transition matrix
    if print_updated_transition_function:
        print("The Updated transition matrix is: ")
        game.print_transition_matrix()


