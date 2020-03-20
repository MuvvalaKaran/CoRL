# file to construct the game and extract admissible strategies
from src.two_player_game.GridWorld import GridWorld
import sys

class TwoPlayerGame:

    def __init__(self):
        pass

    @staticmethod
    def print_env_states(gridworld):
        """
        pass instance of gridworld
        :param gridworld:
        :type gridworld: GridWorld
        :return: None

        """

        for s in gridworld.states:
            print("state ({},{},{}) \n".format(s.x, s.y, s.t))

def main():
    # create a env get the states of the env
    # config_file_name = sys.path
    yaml_name = "env"
    config_file_name = (str(sys.path[0]) + "/config/" + yaml_name)
    gm = GridWorld(config_file_name)
    gm.create_states()
    TwoPlayerGame.print_env_states(gm)

if __name__ == "__main__":

   main()
