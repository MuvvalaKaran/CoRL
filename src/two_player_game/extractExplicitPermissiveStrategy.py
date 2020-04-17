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
# print states to x,y mapping flag
print_state_mapping = True


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

    def get_input_output_AP(self):

        # crete slugsfile.slugsin path
        slugsin_file = self.slugsfile_path + ".slugsin"

        # read the file and store the content in file_data
        file_handle = open(slugsin_file, "r")
        file_data = file_handle.read()

        # after the input section add all the input APs
        input_str = re.compile("\[INPUT\]")
        input_ap = []
        # after the output section add all the output APs
        output_str = re.compile("\[OUTPUT\]")
        comment_str = re.compile("^#")
        output_ap = []

        split_txt = file_data.split("\n")

        for il, line in enumerate(split_txt):

            if input_str.match(line) is not None:
                # keep storing variables in the list till you encounter an empty line
                #tmp_line = line
                # ignore line with comment
                tmp_il = il
                while True:

                    tmp_il += 1
                    if comment_str.match(split_txt[tmp_il]) is not None:
                        continue
                    if split_txt[tmp_il] is "":
                        break
                    input_ap.append(split_txt[tmp_il])

                    # il = tmp_il
                    #tmp_line = line[tmp_il]
            if output_str.match(line) is not None:
                # keep storing variables in the list till you encounter an empty line
                tmp_il = il
                while True:
                    tmp_il += 1
                    if comment_str.match(split_txt[tmp_il]) is not None:
                        continue
                    if split_txt[tmp_il] is "":
                        # once done with this lop breal
                        return input_ap, output_ap

                    output_ap.append(split_txt[tmp_il])

        return input_ap, output_ap

    # @staticmethod
    def interpret_strategy_output(self, file_name):
        # get the input and output atomic propositions
        input_ap, output_ap = self.get_input_output_AP()

        # a dictionary mapping from a state to a pair of (x1,y1), (x2,y2) i.e Input AP and Output AP
        State = {}

        if file_name is None:
            file_name = str(get_cwd_path() + "/slugs_file/CoRL_5.txt")
            print(file_name)
        # intialize state counter
        state_counter = 0
        empty_state_counter = 0

        file_handle = open(file_name, "r")
        output = file_handle.read()

        str_states = re.compile("^State\s\d+")
        int_states = re.compile("^[0-9]*$")

        split_txt = output.split("\n")
        for il, line in enumerate(split_txt):

            # count the number state and map each state to its corresponding in (x1, y1), (x2, y2)
            if str_states.match(line) is not None:
                state_encountered = True
                # get state name

                match = str_states.match(line)
                state_name = match.string[match.regs[0][0]: match.regs[0][1]]
                state_counter += 1
                # count states without transitions
                if split_txt[il + 2] is "":
                    empty_state_counter += 1

                input_bit = []
                output_bit = []
                # read the bit encoding of the respective state
                for i_ap, ip_bit_val in zip(input_ap, split_txt[il+1].split(", ")):
                    # split the bit name and val
                    if i_ap == ip_bit_val.split(":")[0]:
                        # if ip_bit_val.split(":")[1] == str(1):
                        input_bit.append(ip_bit_val.split(":")[1])

                for o_ap, ip_bit_val in zip(output_ap, split_txt[il+1].split(", ")[len(input_ap):]):
                    # split the bit name and val
                    if o_ap == ip_bit_val.split(":")[0]:
                        # if ip_bit_val.split(":")[1] == str(1):
                        output_bit.append(ip_bit_val.split(":")[1])

                # (x1, y1) and (x2, y2)
                x1 = int(''.join(input_bit[:2]), 2)
                y1 = int(''.join(input_bit[2:4]), 2)
                x2 = int(''.join(output_bit[:2]), 2)
                y2 = int(''.join(output_bit[2:4]), 2)
                State.update({state_name: {'map': ((x1, y1), (x2, y2))}})

            # create a transition list from state to the all the other states
            state_transits_to = []

            if state_encountered:
                tmp_il = il + 2
                while True:
                    # the child states are mentioned after the bit encoding
                    match = int_states.match(split_txt[tmp_il])
                    tmp_il += 1
                    if match is not None:
                        state_transits_to.append(match.string[match.regs[0][0]: match.regs[0][1]])
                    if split_txt[tmp_il] is "":
                        # once done set flag to flase
                        state_encountered = False

                        # update State dictionary
                        State[state_name].update({'child_nodes' : state_transits_to})
                        break

        if print_state_mapping:
            for k, v in State.items():
                print(f"{k}: {v}")

        return State

    def main(self):
        pass
        # self.convert_to_slugsin()
        # self.convert_slugsin_to_permissive_str()
        # str_info = self.interpret_strategy_output(None)


if __name__ == "__main__":

    slugsfile_path = "slugs_file/CoRL_5"
    permisivestr = PermissiveStrategy(slugsfile_path)
    # permisivestr.main()