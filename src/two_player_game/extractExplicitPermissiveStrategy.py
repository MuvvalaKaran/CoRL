# a script to execute slugs and extract the permissive strategy
import sys

from subprocess import Popen, PIPE
from pathlib import Path
from os import system as system_call
from platform import system as system_name

print_output = True

class PermissiveStrategy:

    def __init__(self, slugsfile_path , structuredfile=True):
        # slugsfile is a structuredslugs file
        self.slugsfile_path = slugsfile_path
        if structuredfile:
            self.slugsfile = self.slugsfile_path + ".structuredslugs"
        self.local_file_path_to_slugs = "../../lib/slugs/"
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

    def get_cwd_path(self):
        return Path.cwd()

    def convert_to_slugsin(self):
        compiler_path = self.get_compiler_file_path()
        slugscmdline = compiler_path + str(" " + self.slugsfile) + " > " + self.slugsfile_path + ".slugsin"

        print(slugscmdline)
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
                print(output)

        except FileNotFoundError as e:
            PermissiveStrategy.eprint("{} : structuredslugs file not found. \n")
            sys.exit(1)

    def get_slugs_link(self):
        curr_path = self.get_cwd_path()
        # slugspath = str(curr_path) + self.local_file_path_to_slugs
        slugspath = self.local_file_path_to_slugs

        return slugspath

    def get_compiler_file_path(self):
        slugspath = self.get_slugs_link()
        compiler_path = slugspath + self.compiler_file_path

        return str(compiler_path)

    def main(self):
        self.convert_to_slugsin()

if __name__ == "__main__":

    slugsfile_path = "slugs_file/CoRL_1"
    permisivestr = PermissiveStrategy(slugsfile_path)
    permisivestr.main()