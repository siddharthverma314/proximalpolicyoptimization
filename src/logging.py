import termcolor as tc
import datetime as dt
import os
from scipy import io

class Logger:
    """Logs data to a directory"""

    TYPE_SCALAR = 's'
    TYPE_ARRAY = 'a'

    def __init__(self, log_path='../logs/'):
        now = dt.datetime.now()
        print(tc.colored(f"Starting log at {now}", 'red'))
        self.directory = log_path + now.strftime("%Y%m%d%H%M%S")
        self.dict = dict()

    @staticmethod
    def __parse(varname):
        return list(filter(lambda x: x, varname.split('/')))

    def create(self, varname, vartype):
        "Create a varname with filesystem-like addressing"
        cur_dict = self.dict
        vars = self.__parse(varname)

        for name in vars[:-1]:
            if name not in cur_dict:
                cur_dict[name] = dict()
            cur_dict = cur_dict[name]

        if vartype == self.TYPE_SCALAR:
            cur_dict[vars[-1]] = None
            print(tc.colored(f"Created scalar {varname}", 'yellow'))
        elif vartype == self.TYPE_ARRAY:
            cur_dict[vars[-1]] = []
            print(tc.colored(f"Created array {varname}", 'yellow'))

    def log(self, varname, val):
        "Must create a variable before logging it"
        print(tc.colored(f"{varname}: {val}", 'green'))
        varname = self.__parse(varname)

        cur_dict = self.dict
        for name in varname[:-1]:
            cur_dict = cur_dict[name]

        if type(cur_dict[varname[-1]]) == list:
            cur_dict[varname[-1]].append(val)
            print(tc.colored(f"Appended {varname}: {val}", 'green'))
        else:
            cur_dict[varname[-1]] = val
            print(tc.colored(f"Set {varname}: {val}", 'green'))

    def save(self):
        os.mkdir(self.directory)
        fp = f"{self.directory}/log"
        io.savemat(fp, self.dict)
        print(tc.colored(f"Saved file at {fp}", 'red'))
