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
print_state_mapping = False
# flag to be set true when you use explicit player representation
explicit_rep = True

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
            # when running from main "../lib/slugs/"
            self.local_file_path_to_slugs = "../lib/slugs/"
            # when running from the learning folder
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
            process = Popen(structuredslugs2slugsin_args, stdout=slugsin_file_handle)
            # wait till the process is finished. Dumping usually takes some time
            process.wait()
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
            process = Popen(slugstopermissivestr_args, stdout=permisivestr_file_handle)
            # wait till the process is finished. Dumping usually takes some time
            process.wait()
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
        x_str = re.compile("^x")
        y_str = re.compile("^y")
        input_ap = []
        # after the output section add all the output APs
        output_str = re.compile("\[OUTPUT\]")
        comment_str = re.compile("^#")
        output_ap = []

        split_txt = file_data.split("\n")

        for il, line in enumerate(split_txt):

            if input_str.match(line) is not None:
                # keep storing variables in the list till you encounter an empty line
                # ignore line with comment
                tmp_il = il
                x_ip_ap = []
                y_ip_ap = []
                while True:
                    tmp_il += 1
                    if comment_str.match(split_txt[tmp_il]) is not None:
                        continue
                    if split_txt[tmp_il] is "":
                        break

                    input_ap.append(split_txt[tmp_il])
                    # check if the input ap belongs to x or y
                    if x_str.match(split_txt[tmp_il]):
                        # input_ap.append(split_txt[tmp_il])
                        x_ip_ap.append(split_txt[tmp_il])
                    if y_str.match(split_txt[tmp_il]):
                        y_ip_ap.append(split_txt[tmp_il])

            if output_str.match(line) is not None:
                # keep storing variables in the list till you encounter an empty line
                tmp_il = il
                x_op_ap = []
                y_op_ap = []
                while True:
                    tmp_il += 1
                    if comment_str.match(split_txt[tmp_il]) is not None:
                        continue
                    if split_txt[tmp_il] is "":
                        # once done with this lop breal
                        return input_ap, output_ap, (x_ip_ap, y_ip_ap), (x_op_ap, y_op_ap)

                    output_ap.append(split_txt[tmp_il])
                    # check if the input ap belongs to x or y
                    if x_str.match(split_txt[tmp_il]):
                        # input_ap.append(split_txt[tmp_il])
                        x_op_ap.append(split_txt[tmp_il])
                    if y_str.match(split_txt[tmp_il]):
                        y_op_ap.append(split_txt[tmp_il])

        return input_ap, output_ap, (x_ip_ap, y_ip_ap), (x_op_ap, y_op_ap)

    # @staticmethod
    def interpret_strategy_output(self, file_name):
        # get the input and output atomic propositions
        input_ap, output_ap, (x_ip_ap, y_ip_ap), (x_op_ap, y_op_ap) = self.get_input_output_AP()

        # length of x and y atomic propositions in input and output
        len_x_ip_ap = len(x_ip_ap)
        len_y_ip_ap = len(y_ip_ap)
        len_x_op_ap = len(x_op_ap)
        len_y_op_ap = len(y_op_ap)

        # a dictionary mapping from a state to a pair of (x1,y1), (x2,y2) i.e Input AP and Output AP
        State = {}

        if file_name is None:
            file_name = str(get_cwd_path() + "/slugs_file/CoRL_5.txt")
            print(file_name)
        else:
            file_name = str(get_cwd_path() + file_name)
        # initialize state counter
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
                # read the bit encoding of the respective state : the formatting is little endian
                for i_ap, ip_bit_val in zip(input_ap, split_txt[il+1].split(", ")):
                    # split the bit name and val
                    if i_ap == ip_bit_val.split(":")[0]:
                        # if ip_bit_val.split(":")[1] == str(1):
                        # input_bit.append(ip_bit_val.split(":")[1])
                        input_bit.insert(0, ip_bit_val.split(":")[1])
                for o_ap, ip_bit_val in zip(output_ap, split_txt[il+1].split(", ")[len(input_ap):]):
                    # split the bit name and val
                    if o_ap == ip_bit_val.split(":")[0]:
                        # if ip_bit_val.split(":")[1] == str(1):
                        # output_bit.append(ip_bit_val.split(":")[1])
                        output_bit.insert(0, ip_bit_val.split(":")[1])
                # (x1, y1) and (x2, y2)
                # y1 = int(''.join(input_bit[:2]), 2)
                # x1 = int(''.join(input_bit[2:4]), 2)
                # y2 = int(''.join(output_bit[:2]), 2)
                # x2 = int(''.join(output_bit[2:4]), 2)
                # if we explicitly represent players in the slugs input file in the [INPUT] section
                if explicit_rep:
                    # sample format we storing the bits into
                    # t@0.0.1:0/1 y1@1:0/1 y1@0.0.3:0/1 x1@1:0/1 x1@0.0.3:0/1
                    p = input_bit[0]
                    y1 = int(''.join(input_bit[1:1+len_y_ip_ap]), 2)
                    x1 = int(''.join(input_bit[1+len_y_ip_ap:1 + len_y_ip_ap + len_x_ip_ap]), 2)

                    # output bit has no player so we start form the start
                    # sample format we are storing the bits into
                    # y2@1:0/1 y2@0.0.3:0/1 x2@1:0/1 x2@0.0.3:0/1
                    y2 = int(''.join(output_bit[:len_y_op_ap]), 2)
                    x2 = int(''.join(output_bit[len_y_op_ap: len_y_op_ap + len_x_op_ap]), 2)
                # if there is no explict player states in [INPUT]section of the slugs input file
                else:
                    # sample format we storing the bits into
                    # y1@1:0/1 y1@0.0.3:0/1 x1@1:0/1 x1@0.0.3:0/1
                    y1 = int(''.join(input_bit[:len_y_ip_ap ]), 2)
                    x1 = int(''.join(input_bit[len_y_ip_ap: len_y_ip_ap + len_x_ip_ap]), 2)

                    # output bit has no player so we start form the start
                    # sample format we are storing the bits into
                    # y2@1:0/1 y2@0.0.3:0/1 x2@1:0/1 x2@0.0.3:0/1
                    y2 = int(''.join(output_bit[:len_y_op_ap]), 2)
                    x2 = int(''.join(output_bit[len_y_op_ap:len_y_op_ap + len_x_op_ap]), 2)
                mapping_dict = {}
                # x1,y1 belong to the env while x2,y2 belong to the controlled robot/system
                mapping_dict.update({'state_xy_map': ((x2, y2), (x1, y1))})
                mapping_dict.update({'state_pos_map': None})
                if explicit_rep:
                    mapping_dict.update({'player': p})
                # mapping_dict = {{'state_xy_map': ((x1, y1), (x2, y2))}, {'state_pos_map': None}}
                # State.update({state_name: {'state_xy_map': ((x1, y1), (x2, y2))}})
                # State.update({state_name: {'state_pos_map': None}})
                State.update({state_name: mapping_dict})

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
                        # once done set flag to false
                        state_encountered = False

                        # update State dictionary
                        State[state_name].update({'child_nodes' : state_transits_to})
                        break

        if print_state_mapping:
            self.print_str_output(State)

        return State

    def print_str_output(self, str):
        for k, v in str.items():
            print(f"{k}: {v}")

    def main(self):
        pass

        # self.convert_to_slugsin()
        # self.convert_slugsin_to_permissive_str()
        # str_info = self.interpret_strategy_output(None)


if __name__ == "__main__":

    slugsfile_path = "slugs_file/CoRL_5"
    permisivestr = PermissiveStrategy(slugsfile_path)
    # permisivestr.main()