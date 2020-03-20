# file to construct the game and extract admissible strategies
from src.two_player_game.GridWorld import GridWorld
import sys
import random

random.seed(1)

class TwoPlayerGame:

    def __init__(self, gm):
        self.gm = gm

    @staticmethod
    def print_env_states(gridworld):
        """
        pass instance of gridworld
        :param gridworld:
        :type gridworld: GridWorld
        :return: None

        """

        for s_r in gridworld.states:
            for s_c in s_r:
                print("state ({},{},{}) \n".format(s_c.x, s_c.y, s_c.t))
        # print(list(gridworld.states))

    def set_state_players(self):

        for row_state in self.gm.states:
            for col_state in row_state:
                GridWorld.set_state_player(col_state, random.randint(0, 1), self.gm)


def main():
    # create a env get the states of the env
    # config_file_name = sys.path
    yaml_name = "env"
    config_file_name = (str(sys.path[0]) + "/config/" + yaml_name)
    # instance of girdworld object
    gm = GridWorld(config_file_name)
    gm.create_states()
    game = TwoPlayerGame(gm)
    game.set_state_players()
    game.print_env_states(gm)


if __name__ == "__main__":

   main()
