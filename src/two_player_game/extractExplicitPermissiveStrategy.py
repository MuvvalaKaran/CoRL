# a script to execute slugs and extract the permissive strategy
import sys
import os
import time
import re

from graphviz import Source
from subprocess import Popen, PIPE
from pathlib import Path
from os import system as system_call
from platform import system as system_name

# flag to print the output in the terminal
print_output = False
#flag to print the output from the slugs code using --extractExplcitPermissiveStrategy flag
print_str = False


def get_cwd_path():
    # return Path.cwd()
    return os.path.dirname(os.path.realpath(__file__))

class PermissiveStrategy:

    def __init__(self, slugsfile_path , structuredfile=True, run_local=True):
        # slugsfile is a structuredslugs file
        self.slugsfile_path = slugsfile_path
        if structuredfile:
            self.slugsfile = self.slugsfile_path + ".structuredslugs"
        if run_local:
            self.local_file_path_to_slugs = "../../lib/slugs/"
        else:
            self.local_file_path_to_slugs = "../lib/slugs/"
        self.compiler_file_path = "tools/StructuredSlugsParser/compiler.py"

    @staticmethod
    def clear_shell_screen():
        """
        clears the shell screen
        :return: None
        :rtype: None
        """
        cmd = "cls" if system_name().lower() == "windows" else "clear"
        system_call(cmd)

    @staticmethod
    def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

    # def get_cwd_path(self):
    #     # return Path.cwd()
    #     return os.path.dirname(os.path.realpath(__file__))

    def plot_graphs_from_dot_files(self, file_name, view=False):
        s = Source.from_file(file_name)
        if view:
            s.view()
            time.sleep(1)


    def convert_to_slugsin(self):
        compiler_path = self.get_compiler_file_path()
        # open a terminal and execute the command
        try:
            structuredslugs2slugsin_args = [str(compiler_path), str(self.slugsfile)]

            # create a file handle
            slugsin_file_handle = open(str(self.slugsfile_path + ".slugsin"), "w+")

            # write output to a file
            Popen(structuredslugs2slugsin_args, stdout=slugsin_file_handle)
            if print_output:
                process = Popen(structuredslugs2slugsin_args, stdout=PIPE)
                (output, err) = process.communicate()
                output = output.decode('utf-8')
                print("Print Slugsin file compiled from the structuredslugs file")
                print(output)

        except FileNotFoundError as e:
            PermissiveStrategy.eprint(f"{str(self.slugsfile)} : structuredslugs file not found. \n")
            sys.exit(1)

    def convert_slugsin_to_permissive_str(self):
        slugslink = self.get_slugs_link()
        # slugsin_file_handle = open(str(self.slugsfile_path + ".slugsin"), "r")
        # use the permissive flag to compute a permissible strategy
        addition_parameter = "--extractExplicitPermissiveStrategy"

        permisivestr_file_handle = open(str(self.slugsfile_path + ".txt"), "w+")

        try:
            slugstopermissivestr_args = [str(slugslink + "src/slugs"),
                                         addition_parameter,
                                         str(self.slugsfile_path + ".slugsin")]
            Popen(slugstopermissivestr_args, stdout=permisivestr_file_handle)
            if print_str:
                process = Popen(slugstopermissivestr_args, stdout=PIPE)
                (output, err) = process.communicate()
                output = output.decode("utf-8")
                print("Printing Permissive Str computed from slugs ")
                print(output)


        except FileNotFoundError as e:
            PermissiveStrategy.eprint(f"{self.slugsfile_path + str('.slugsin')} : slugsin file not found ")
            sys.exit(1)

    def get_slugs_link(self):
        # curr_path = self.get_cwd_path()
        # slugspath = str(curr_path) + self.local_file_path_to_slugs
        slugspath = self.local_file_path_to_slugs

        return slugspath

    def get_compiler_file_path(self):
        slugspath = self.get_slugs_link()
        compiler_path = slugspath + self.compiler_file_path

        return str(compiler_path)

    # added helper methods to read the strategy txt file to count the number of states,
    # no. of states without any transitions
    @staticmethod
    def interpret_strategy_output(file_name):
        # print("This is the name of the script", sys.argv[0])
        # print("Number of arguments", len(sys.argv))
        # print("The arguments are :", str(sys.argv))

        if file_name is None:
            file_name = str(get_cwd_path() + "/slugs_file/CoRL_3.txt")
            print(file_name)
        # intialize state counter
        state_counter = 0
        player0_counter = 0
        player1_counter = 0
        empty_state_counter = 0

        file_handle = open(file_name, "r")
        output = file_handle.read()

        str_states = re.compile("^State")
        str_player0 = re.compile("player0:1")
        str_player1 = re.compile("player1:1")
        split_txt = output.split("\n")
        for il, line in enumerate(split_txt):
            # count the number states
            if str_states.match(line) is not None:
                state_counter += 1
                # print(line)
                # count states without transitions
                if split_txt[il + 2] is "":
                    empty_state_counter += 1

            # if str_player0.match(line) is not None:
            if re.search("player0:1", line):
                player0_counter += 1

            # if str_player1.match(line) is not None:
            if re.search("player1:1", line):
                player1_counter += 1

        print(state_counter)
        print(empty_state_counter)
        print(player0_counter)
        print(player1_counter)

            # count the number of states without any transitions

            # count the number of states with some transitions to other states



    def main(self):
        # self.convert_to_slugsin()
        # self.convert_slugsin_to_permissive_str()
        PermissiveStrategy.clear_shell_screen()
        # PermissiveStrategy.interpret_strategy_output(sys.argv[1])
        if sys.argv[1] is None:
            PermissiveStrategy.interpret_strategy_output(None)
        else:
            PermissiveStrategy.interpret_strategy_output(sys.argv[1])

if __name__ == "__main__":

    slugsfile_path = "slugs_file/CoRL_1"
    permisivestr = PermissiveStrategy(slugsfile_path)
    permisivestr.main()