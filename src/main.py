# parent script file
# 1. Construct a game G and Extract a permissive str using slugs
# 2. Check if the specification is realizable and if so apply the str to G to compute C_hat
# 3. Compute the most optimal str that maximizes the reward while also satisfying the safety and liveness guarantees
# 4. Map the optimal str back to G

from src.two_player_game import TwoPlayerGame
from src.two_player_game import extractExplicitPermissiveStrategy

# flag to plot BDD disgram we get from the slugs file
plot_graphs = False

def construct_game():
    # run the main function to build all the stuff in there
    game = TwoPlayerGame.main()
    # print(game.get_atomic_propositons())
    # print(game.get_initial_state())
    # print(game.get_players_data())
    return game

def extract_permissive_str():
    slugsfile_path = "two_player_game/slugs_file/CoRL_5"
    str = extractExplicitPermissiveStrategy.PermissiveStrategy(slugsfile_path, run_local=False)
    # str.main()
    env_str = None
    system_str = str.interpret_strategy_output(None)
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
    Pos = x_length*x + y
    return Pos

if __name__ == "__main__":
    # construct_game()
    game = construct_game()
    x_length = game.env.env_data["env_size"]["n"]
    y_length = game.env.env_data["env_size"]["m"]
    sys_str, env_str = extract_permissive_str()
    a_c = game.get_controlled_actions()
    # create a mapping from th state we get form the strategy and the state we have in our code
    # s = sys_str['State 1']['map']
    # print(s)

    # create a tmp_state_tuple
    for k, v in sys_str.items():
        xy_sys = v['state_xy_map'][0]
        xy_env = v['state_xy_map'][1]
        pos_sys = create_xy_pos_mapping(xy_sys[0], xy_sys[1], x_length, y_length)
        pos_env = create_xy_pos_mapping(xy_env[0], xy_env[1], x_length, y_length)
        t = 1
        # print(f"{k} : {xy_sys}, {xy_env}0 : ({pos_env}, {pos_env}, {t})")

        # update the state_pos mapping variable in sys_str
        sys_str[k]['state_pos_map'] = (pos_sys, pos_env, t)
        print(k, sys_str[k]['state_pos_map'])

        # game.set_transition_function(game.S[pos_sys][pos_env][t], )



