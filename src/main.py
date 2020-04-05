# parent script file
# 1. Construct a game G and Extract a permissive str using slugs
# 2. Check if the specification is realizable and if so apply the str to G to compute C_hat
# 3. Compute the most optimal str that maximizes the reward while also satisfying the safety and liveness guarantees
# 4. Map the optimal str back to G

from src.two_player_game import TwoPlayerGame
from src.two_player_game import extractExplicitPermissiveStrategy

def construct_game():
    # run the main function to build all the stuff in there
    game = TwoPlayerGame.main()
    # print(game.get_atomic_propositons())
    # print(game.get_initial_state())
    # print(game.get_players_data())

def extract_permissive_str():
    slugsfile_path = "two_player_game/slugs_file/CoRL_1"
    str = extractExplicitPermissiveStrategy.PermissiveStrategy(slugsfile_path, run_local=False)
    str.main()
    for i in range(50):
        file_name = f"two_player_game/graphs/state{i}.dot"
        str.plot_graphs_from_dot_files(file_name, view=True)

if __name__ == "__main__":
    # construct_game()
    extract_permissive_str()
