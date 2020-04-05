# a script to execute slugs and extract the permissive strategy
import sys
import os
import time

from graphviz import Source
from subprocess import Popen, PIPE
from pathlib import Path
from os import system as system_call
from platform import system as system_name

# flag to print the output in the terminal
print_output = False
#flag to print the output from the slugs code using --extractExplcitPermissiveStrategy flag
print_str = False


def get_cwd_path(self):
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

    def main(self):
        self.convert_to_slugsin()
        self.convert_slugsin_to_permissive_str()


if __name__ == "__main__":

    slugsfile_path = "slugs_file/CoRL_1"
    permisivestr = PermissiveStrategy(slugsfile_path)
    permisivestr.main()